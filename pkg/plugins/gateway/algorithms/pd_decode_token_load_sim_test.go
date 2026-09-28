/*
Copyright 2024 The Aibrix Team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package routingalgorithms

import (
	"container/heap"
	"fmt"
	"math"
	"math/rand"
	"net/http"
	"os"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"github.com/vllm-project/aibrix/pkg/cache"
	"github.com/vllm-project/aibrix/pkg/constants"
	"github.com/vllm-project/aibrix/pkg/metrics"
	"github.com/vllm-project/aibrix/pkg/plugins/gateway/algorithms/pd"
	"github.com/vllm-project/aibrix/pkg/plugins/gateway/algorithms/pd/prefill"
	"github.com/vllm-project/aibrix/pkg/plugins/gateway/algorithms/pd/selector"
	"github.com/vllm-project/aibrix/pkg/types"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// This file is a discrete-event simulation of decode pod selection, not a unit
// test: it drives the real filterPrefillDecodePods against a simple KV model
// and prints a comparison table. It only runs with AIBRIX_PD_DECODE_SIM=1.
//
// Model, per decode pod:
//   - a fixed KV capacity in tokens;
//   - a request holds prompt_tokens of KV from routing until it completes, plus
//     output tokens that grow linearly over its decode time (the ledger does not
//     see this growth);
//   - RealtimeNumRequestsRunning is updated on every routing and completion, as
//     the gateway does; KVCacheUsagePerc is refreshed from the true usage only
//     once per scrape interval.
//
// An admission that takes a pod past its KV capacity is counted as an overflow,
// a stand-in for the engine preempting. The simulation does not model the
// engine's reaction (queueing, recompute), prefill, or network time.

const simModel = "sim-model"

type simConfig struct {
	name           string
	decoders       int
	kvCapacity     float64 // tokens per decode pod
	ratePerDecoder float64 // Poisson arrivals per second per decode pod
	longFraction   float64 // share of arrivals that are long prompts
	shortTokens    [2]int  // uniform range
	longTokens     [2]int  // uniform range
	outputTokens   int
	decodeTokRate  float64 // output tokens per second per request
	burstEvery     time.Duration
	burstSize      int // long requests per burst, per decode pod
	burstSpread    time.Duration
	duration       time.Duration
	scrape         time.Duration
	// requestCost overrides the per-request cost charged to the ledger;
	// negative keeps the default.
	requestCost float64
	// noCountFastPath raises the decode request-count spread threshold so the
	// request-count fast path never fires.
	noCountFastPath bool
	// estimateNoise is the sigma of a lognormal factor between the router's
	// prompt estimate (body/4) and the KV the request really takes: 0 makes
	// the estimate exact.
	estimateNoise float64
}

type simResult struct {
	routed, overflows int
	peakMax           float64 // highest peak KV fraction across pods
	meanSpread        float64 // time-averaged (max-min) KV fraction across pods
	meanUtil          float64 // time-averaged mean KV fraction across pods
	// meanExcess is the time-averaged KV demand above capacity, summed over
	// pods, as a fraction of one pod's capacity: how much KV did not fit.
	meanExcess float64
}

type simReq struct {
	id         string
	pod        int
	prompt     float64
	output     float64
	start, end time.Duration
}

type endHeap []*simReq

func (h endHeap) Len() int           { return len(h) }
func (h endHeap) Less(i, j int) bool { return h[i].end < h[j].end }
func (h endHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *endHeap) Push(x any)        { *h = append(*h, x.(*simReq)) }
func (h *endHeap) Pop() any          { old := *h; n := len(old); x := old[n-1]; *h = old[:n-1]; return x }
func (h endHeap) peek() *simReq      { return h[0] }

func simPod(name, role string) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: "sim",
			Labels: map[string]string{
				PDRoleSetIdentifier:      "sim-rs",
				PDRoleIdentifier:         role,
				constants.ModelLabelName: simModel,
			},
		},
		Status: v1.PodStatus{
			PodIP:      "127.0.0.1",
			Conditions: []v1.PodCondition{{Type: v1.PodReady, Status: v1.ConditionTrue}},
		},
	}
}

// simArrivals generates arrival times and prompt sizes for cfg.
func simArrivals(cfg simConfig, rng *rand.Rand) []struct {
	at     time.Duration
	prompt int
} {
	uniform := func(r [2]int) int { return r[0] + rng.Intn(r[1]-r[0]+1) }
	var out []struct {
		at     time.Duration
		prompt int
	}
	rate := cfg.ratePerDecoder * float64(cfg.decoders)
	for t := time.Duration(0); ; {
		t += time.Duration(rng.ExpFloat64() / rate * float64(time.Second))
		if t >= cfg.duration {
			break
		}
		p := uniform(cfg.shortTokens)
		if rng.Float64() < cfg.longFraction {
			p = uniform(cfg.longTokens)
		}
		out = append(out, struct {
			at     time.Duration
			prompt int
		}{t, p})
	}
	if cfg.burstEvery > 0 {
		for b := cfg.burstEvery; b < cfg.duration; b += cfg.burstEvery {
			for i := 0; i < cfg.burstSize*cfg.decoders; i++ {
				at := b + time.Duration(rng.Int63n(int64(cfg.burstSpread)+1))
				out = append(out, struct {
					at     time.Duration
					prompt int
				}{at, uniform(cfg.longTokens)})
			}
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].at < out[j].at })
	return out
}

func runDecodeSim(t *testing.T, cfg simConfig, policy pd.DecodeScorePolicy, seed int64) simResult {
	t.Helper()
	rng := rand.New(rand.NewSource(seed))

	pods := []*v1.Pod{simPod("prefill-0", "prefill")}
	decodes := make([]*v1.Pod, cfg.decoders)
	index := map[string]int{}
	for i := range decodes {
		decodes[i] = simPod(fmt.Sprintf("decode-%d", i), "decode")
		index[decodes[i].Name] = i
		pods = append(pods, decodes[i])
	}

	tokenCfg := pd.TokenLoadConfig{KVWeight: pd.DefaultTokenLoadKVWeight, RequestCost: pd.DefaultTokenLoadRequestCost}
	if cfg.requestCost >= 0 {
		tokenCfg.RequestCost = cfg.requestCost
	}
	installTokenLoadDefaults(t, tokenCfg)
	if cfg.noCountFastPath {
		restore := types.DefaultRoutingOverrides()
		next := *restore
		next.PD.Spreads.DecodeLoadImbalanceMinSpread = math.MaxFloat64
		types.SetDefaultRoutingOverrides(&next)
		t.Cleanup(func() { types.SetDefaultRoutingOverrides(restore) })
	}
	tokenLoad := pd.NewTokenLoadTrackerWithConfig(tokenCfg)
	t.Cleanup(tokenLoad.Close)
	store := cache.NewWithPodsForTest(pods, simModel)
	r := &pdRouter{
		cache:                 store,
		prefillPolicy:         pd.NewLeastRequestPrefillPolicy(),
		decodePolicy:          policy,
		prefillRequestTracker: pd.NewPrefillRequestTracker(),
		pendingDecodeTracker:  pd.NewPendingDecodeTracker(),
		tokenLoadTracker:      tokenLoad,
		httpClient:            &http.Client{},
		prefixUpdateCh:        make(chan prefixUpdateJob, 1024),
		selectionCounts:       map[string]int64{},
	}
	r.podSelector = selector.NewDefaultSelector(r.filterPrefillDecodePods)
	r.prefillExecutor = prefill.NewDefaultExecutor(r.httpClient, r.prefillRequestTracker, prefill.WithTokenLoadTracker(tokenLoad))

	running := make([]int, cfg.decoders)
	active := make([][]*simReq, cfg.decoders)
	kvAt := func(pod int, now time.Duration) float64 {
		total := 0.0
		for _, q := range active[pod] {
			frac := float64(now-q.start) / float64(q.end-q.start)
			total += q.prompt + q.output*math.Min(1, math.Max(0, frac))
		}
		return total
	}
	publishRunning := func(pod int) {
		cache.InitWithPodsMetrics(store, map[string]map[string]metrics.MetricValue{
			decodes[pod].Name: {metrics.RealtimeNumRequestsRunning: &metrics.SimpleMetricValue{Value: float64(running[pod])}},
		})
	}
	scrape := func(now time.Duration) {
		m := map[string]map[string]metrics.MetricValue{}
		for i, d := range decodes {
			m[d.Name] = map[string]metrics.MetricValue{
				metrics.KVCacheUsagePerc:                &metrics.SimpleMetricValue{Value: math.Min(1, kvAt(i, now)/cfg.kvCapacity)},
				metrics.AvgGenerationThroughputToksPerS: &metrics.SimpleMetricValue{Value: 1000},
			}
		}
		cache.InitWithPodsModelMetrics(store, m)
	}
	for i := range decodes {
		publishRunning(i)
	}
	scrape(0)

	var ends endHeap
	complete := func(until time.Duration) {
		for ends.Len() > 0 && ends.peek().end <= until {
			q := heap.Pop(&ends).(*simReq)
			running[q.pod]--
			for i, a := range active[q.pod] {
				if a == q {
					active[q.pod] = append(active[q.pod][:i], active[q.pod][i+1:]...)
					break
				}
			}
			r.DoneRequestCount(nil, q.id, simModel, 0)
			publishRunning(q.pod)
		}
	}

	var res simResult
	peak := make([]float64, cfg.decoders)
	nextScrape := cfg.scrape
	const sampleEvery = 100 * time.Millisecond
	nextSample := sampleEvery
	var spreadSum, utilSum, excessSum float64
	var spreadSamples int
	sample := func(now time.Duration) {
		lo, hi := math.Inf(1), math.Inf(-1)
		for i := range decodes {
			u := kvAt(i, now) / cfg.kvCapacity
			peak[i] = math.Max(peak[i], u)
			lo, hi = math.Min(lo, u), math.Max(hi, u)
			utilSum += u / float64(cfg.decoders)
			excessSum += math.Max(0, u-1)
		}
		spreadSum += hi - lo
		spreadSamples++
	}

	for n, a := range simArrivals(cfg, rng) {
		// Scrapes (and spread samples) that fall before this arrival.
		for nextScrape <= a.at || nextSample <= a.at {
			if nextSample <= nextScrape {
				complete(nextSample)
				sample(nextSample)
				nextSample += sampleEvery
				continue
			}
			complete(nextScrape)
			scrape(nextScrape)
			nextScrape += cfg.scrape
		}
		complete(a.at)

		id := fmt.Sprintf("r%d", n)
		ctx := tokenLoadRequest(t, id, a.prompt*4)
		ctx.Model = simModel
		_, decode, err := r.filterPrefillDecodePods(ctx, pods)
		require.NoError(t, err)
		// Route drops these once selection and the prefill call are done.
		r.pendingDecodeTracker.RemovePendingDecode(id)
		r.prefillRequestTracker.RemovePrefillRequest(id)

		pod := index[decode.Name]
		prompt := float64(a.prompt)
		if cfg.estimateNoise > 0 {
			prompt *= math.Exp(rng.NormFloat64()*cfg.estimateNoise - cfg.estimateNoise*cfg.estimateNoise/2)
		}
		if kvAt(pod, a.at)+prompt > cfg.kvCapacity {
			res.overflows++
		}
		out := float64(cfg.outputTokens)
		q := &simReq{id: id, pod: pod, prompt: prompt, output: out, start: a.at,
			end: a.at + time.Duration(out/cfg.decodeTokRate*float64(time.Second))}
		active[pod] = append(active[pod], q)
		heap.Push(&ends, q)
		running[pod]++
		publishRunning(pod)
		res.routed++
	}
	for _, p := range peak {
		res.peakMax = math.Max(res.peakMax, p)
	}
	if spreadSamples > 0 {
		res.meanSpread = spreadSum / float64(spreadSamples)
		res.meanUtil = utilSum / float64(spreadSamples)
		res.meanExcess = excessSum / float64(spreadSamples)
	}
	return res
}

func TestPDRouter_DecodeTokenLoadSimulation(t *testing.T) {
	if os.Getenv("AIBRIX_PD_DECODE_SIM") != "1" {
		t.Skip("simulation; set AIBRIX_PD_DECODE_SIM=1 to run")
	}
	base := simConfig{
		kvCapacity:     160_000,
		ratePerDecoder: 3.0,
		longFraction:   0.10,
		shortTokens:    [2]int{200, 2000},
		longTokens:     [2]int{16_000, 32_000},
		outputTokens:   256,
		decodeTokRate:  30,
		duration:       120 * time.Second,
		scrape:         time.Second,
		requestCost:    -1,
	}
	var scenarios []simConfig
	for _, n := range []int{2, 4, 8, 16, 32} {
		steady := base
		steady.name, steady.decoders = "steady", n
		scenarios = append(scenarios, steady)
		bursty := base
		bursty.name, bursty.decoders = "bursty", n
		bursty.ratePerDecoder = 2.0
		bursty.burstEvery, bursty.burstSize, bursty.burstSpread = 10*time.Second, 2, 300*time.Millisecond
		scenarios = append(scenarios, bursty)
	}
	policies := []pd.DecodeScorePolicy{pd.LoadBalancingDecodePolicy{}, pd.LeastRequestDecodePolicy{}, pd.TokenLoadDecodePolicy{}}
	const seeds = 20

	var b strings.Builder
	fmt.Fprintf(&b, "\n%-7s %4s  %-14s %8s %7s %17s %17s\n", "load", "N", "policy", "routed", "mean KV", "overflows (±sd)", "excess KV% (±sd)")
	for _, sc := range scenarios {
		for _, p := range policies {
			var routed, util float64
			overflows := make([]float64, 0, seeds)
			excess := make([]float64, 0, seeds)
			for s := int64(1); s <= seeds; s++ {
				r := runDecodeSim(t, sc, p, s)
				routed += float64(r.routed)
				util += r.meanUtil
				overflows = append(overflows, float64(r.overflows))
				excess = append(excess, 100*r.meanExcess)
			}
			mean, sd := meanStddev(overflows)
			emean, esd := meanStddev(excess)
			fmt.Fprintf(&b, "%-7s %4d  %-14s %8.0f %6.0f%% %9.1f (±%5.1f) %9.2f (±%5.2f)\n", sc.name, sc.decoders, p.Name(),
				routed/seeds, 100*util/seeds, mean, sd, emean, esd)
		}
	}

	// Sensitivity to the scrape interval: if the gap comes from the stale KV
	// signal, it should shrink as the scrape gets faster.
	sens := base
	sens.name, sens.decoders = "bursty", 8
	sens.ratePerDecoder = 2.0
	sens.burstEvery, sens.burstSize, sens.burstSpread = 10*time.Second, 2, 300*time.Millisecond
	fmt.Fprintf(&b, "\nscrape interval sensitivity (bursty, N=8)\n%-8s  %-14s %17s\n", "scrape", "policy", "overflows (±sd)")
	for _, interval := range []time.Duration{250 * time.Millisecond, time.Second, 2 * time.Second} {
		sens.scrape = interval
		for _, p := range []pd.DecodeScorePolicy{pd.LoadBalancingDecodePolicy{}, pd.TokenLoadDecodePolicy{}} {
			overflows := make([]float64, 0, seeds)
			for s := int64(1); s <= seeds; s++ {
				overflows = append(overflows, float64(runDecodeSim(t, sens, p, s).overflows))
			}
			mean, sd := meanStddev(overflows)
			fmt.Fprintf(&b, "%-8s  %-14s %9.1f (±%5.1f)\n", interval, p.Name(), mean, sd)
		}
	}
	t.Log(b.String())
}

func meanStddev(xs []float64) (float64, float64) {
	mean := 0.0
	for _, x := range xs {
		mean += x
	}
	mean /= float64(len(xs))
	variance := 0.0
	for _, x := range xs {
		variance += (x - mean) * (x - mean)
	}
	return mean, math.Sqrt(variance / float64(len(xs)-1))
}

// TestPDRouter_DecodeTokenLoadUtilizationSweep varies the arrival rate at N=8 to
// find the range of decode KV utilization over which the policies differ.
func TestPDRouter_DecodeTokenLoadUtilizationSweep(t *testing.T) {
	if os.Getenv("AIBRIX_PD_DECODE_SIM") != "1" {
		t.Skip("simulation; set AIBRIX_PD_DECODE_SIM=1 to run")
	}
	const seeds = 20
	base := simConfig{
		decoders:      8,
		kvCapacity:    160_000,
		longFraction:  0.10,
		shortTokens:   [2]int{200, 2000},
		longTokens:    [2]int{16_000, 32_000},
		outputTokens:  256,
		decodeTokRate: 30,
		duration:      120 * time.Second,
		scrape:        time.Second,
	}
	base.requestCost = -1
	if v := os.Getenv("AIBRIX_PD_DECODE_SIM_REQUEST_COST"); v != "" {
		_, err := fmt.Sscanf(v, "%g", &base.requestCost)
		require.NoError(t, err)
	}
	base.noCountFastPath = os.Getenv("AIBRIX_PD_DECODE_SIM_NO_FASTPATH") == "1"
	if v := os.Getenv("AIBRIX_PD_DECODE_SIM_ESTIMATE_NOISE"); v != "" {
		_, err := fmt.Sscanf(v, "%g", &base.estimateNoise)
		require.NoError(t, err)
	}
	var b strings.Builder
	fmt.Fprintf(&b, "\nestimate noise sigma: %g\nno count fast path: %v\nrequest cost %g (negative: default)\n%-7s %5s %7s  %17s %17s  %17s %17s\n", base.estimateNoise, base.noCountFastPath, base.requestCost, "load", "rate", "mean KV", "LB overflows", "TL overflows", "LB excess KV%", "TL excess KV%")
	for _, load := range []string{"steady", "bursty"} {
		for _, rate := range []float64{1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5} {
			sc := base
			sc.name, sc.ratePerDecoder = load, rate
			if load == "bursty" {
				sc.burstEvery, sc.burstSize, sc.burstSpread = 10*time.Second, 2, 300*time.Millisecond
			}
			var util float64
			cols := make([]string, 0, 4)
			for _, p := range []pd.DecodeScorePolicy{pd.LoadBalancingDecodePolicy{}, pd.TokenLoadDecodePolicy{}} {
				overflows := make([]float64, 0, seeds)
				excess := make([]float64, 0, seeds)
				for s := int64(1); s <= seeds; s++ {
					r := runDecodeSim(t, sc, p, s)
					overflows = append(overflows, float64(r.overflows))
					excess = append(excess, 100*r.meanExcess)
					util += r.meanUtil
				}
				mean, sd := meanStddev(overflows)
				emean, esd := meanStddev(excess)
				cols = append(cols, fmt.Sprintf("%7.1f (±%5.1f)", mean, sd), fmt.Sprintf("%6.2f (±%5.2f)", emean, esd))
			}
			fmt.Fprintf(&b, "%-7s %5.1f %6.0f%%  %17s %17s  %17s %17s\n", load, rate, 100*util/(2*seeds), cols[0], cols[2], cols[1], cols[3])
		}
	}
	t.Log(b.String())
}
