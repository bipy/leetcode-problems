package algo

import (
	"github.com/stretchr/testify/assert"
	"math/rand"
	"sort"
	"testing"
)

func TestTopK(t *testing.T) {
	k := 328
	ori := make([]*Item, 10000)
	for i := range ori {
		ori[i] = &Item{value: rand.Intn(10000)}
	}
	sorted := make([]*Item, len(ori))
	for i := range sorted {
		sorted[i] = &Item{value: ori[i].value}
	}
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].value.(int) < sorted[j].value.(int)
	})
	topk := TopK(k, ori, func(i, j int) bool {
		return ori[i].value.(int) < ori[j].value.(int)
	})
	sort.Slice(topk, func(i, j int) bool {
		return topk[i].value.(int) < topk[j].value.(int)
	})
	assert.Equal(t, sorted[:k], topk)
}
