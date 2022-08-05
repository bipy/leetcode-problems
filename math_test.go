package leetcode

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestIsPrime(t *testing.T) {
	var p []int
	for i := 1; i < 1001; i++ {
		if IsPrime(i) {
			p = append(p, i)
		}
	}
	assert.Equal(t, Primes, p)
}

func TestGCD(t *testing.T) {
	assert.Equal(t, 6, GCD(30, 36))
}

func TestLCM(t *testing.T) {
	assert.Equal(t, 12, LCM(4, 6))
}

func TestSummarizingInt(t *testing.T) {
	assert.Equal(t, 10, SummarizingInt([]int{1, 2, 3, 4}))
}

func TestSummarizingFloat(t *testing.T) {
	assert.Equal(t, 7.2, SummarizingFloat([]float64{1.5, 2.5, 3.2}))
}

func TestC(t *testing.T) {
	assert.Equal(t, 6, C(2, 4))
}
