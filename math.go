package leetcode

import "math"

// Primes 1000 以内质数
var Primes = []int{1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997}

// IsPrime 是否质数
func IsPrime(x int) bool {
	for i := 2; i*i <= x; i++ {
		if x%i == 0 {
			return false
		}
	}
	return true
}

// GCD 最大公因数
func GCD(a, b int) int {
	for a != 0 {
		a, b = b%a, a
	}
	return b
}

// LCM 最小公倍数
func LCM(a, b int) int {
	return a / GCD(a, b) * b
}

// SummarizingInt 整数求和
func SummarizingInt(arr []int) (sum int) {
	for i := range arr {
		sum += arr[i]
	}
	return
}

// SummarizingFloat 浮点数求和
func SummarizingFloat(arr []float64) (sum float64) {
	for i := range arr {
		sum += arr[i]
	}
	return
}

// AverageInt 整数平均值
func AverageInt(arr []int) float64 {
	return float64(SummarizingInt(arr)) / float64(len(arr))
}

// AverageFloat 浮点数平均值
func AverageFloat(arr []float64) float64 {
	return SummarizingFloat(arr) / float64(len(arr))
}

// IsInt 是否整数
func IsInt(x float64) bool {
	const eps = 1e-8
	return math.Abs(x-math.Round(x)) < eps
}

// FastP 矩阵快速幂
func FastP(matrix [][]int) int {
	return 0
}
