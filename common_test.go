package leetcode

import (
	"fmt"
	"github.com/stretchr/testify/assert"
	"testing"
)

func Test_isPrime(t *testing.T) {
	for i := 1; i < 1001; i++ {
		if isPrime(i) {
			fmt.Printf("%d, ", i)
		}
	}
	for i := range primes {
		if !isPrime(primes[i]) {
			fmt.Println(primes[i])
		}
	}
}

func Test_maxOf(t *testing.T) {
	assert.Equal(t, 10, maxOf(1, 2, 4, 8, 10, 4, 1))
	assert.Equal(t, 0x7fffffff, maxOf(1, 2, 4, 8, 10, 4, 1, 0x7fffffff))
}

func Test_minOf(t *testing.T) {
	assert.Equal(t, 1, minOf(1, 2, 4, 8, 10, 4, 1))
	assert.Equal(t, -1<<31, minOf(1, 2, 4, 8, 10, 4, 1, -1<<31))
}
