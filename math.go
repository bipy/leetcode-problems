package leetcode

import (
	"math"
)

type Frac struct {
	up, down int
}

func (f *Frac) Add(x Frac) {
	lcm := LCM(f.down, x.down)
	x.up *= lcm / x.down
	f.up *= lcm / f.down
	f.up += x.up
	f.down = lcm
	f.Simplify()
}

func (f *Frac) Sub(x Frac) {
	lcm := LCM(f.down, x.down)
	x.up *= lcm / x.down
	f.up *= lcm / f.down
	f.up -= x.up
	f.down = lcm
	f.Simplify()
}

func (f *Frac) Mul(x Frac) {
	f.up *= x.up
	f.down *= x.down
	f.Simplify()
}

func (f *Frac) Div(x Frac) {
	if x.up < 0 {
		x.up, x.down = -x.down, -x.up
	} else {
		x.up, x.down = x.down, x.up
	}
	f.Mul(x)
}

func (f *Frac) Simplify() {
	if f.up == 0 {
		f.down = 1
	} else {
		gcd := GCD(abs(f.up), f.down)
		f.up /= gcd
		f.down /= gcd
	}
}

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

// Ceil 向上取整除
func Ceil(a, b int) int {
	return (a + b - 1) / b
}

// Transpose 矩阵转置
func Transpose(a [][]int) [][]int {
	n, m := len(a), len(a[0])
	b := make([][]int, m)
	for i := range b {
		b[i] = make([]int, n)
		for j, r := range a {
			b[i][j] = r[i]
		}
	}
	return b
}

// Pow 模幂
func Pow(x, n, mod int) int {
	x %= mod
	res := 1 % mod
	for ; n > 0; n >>= 1 {
		if n&1 == 1 {
			res = res * x % mod
		}
		x = x * x % mod
	}
	return res
}

// 平面距离的平方
func getDistSq(dx, dy int) int {
	return dx*dx + dy*dy
}

// 阶乘值
var fracNums = []int{1}

// 阶乘
func frac(a, b int) int {
	for i := len(fracNums); i <= b; i++ {
		fracNums = append(fracNums, fracNums[i-1]*i)
	}
	return fracNums[b] / fracNums[a-1]
}

// C 组合数
func C(a, b int) int {
	if a > b {
		a, b = b, a
	}
	return frac(b-a+1, b) / frac(1, a)
}

// A 排列数
func A(a, b int) int {
	if a > b {
		a, b = b, a
	}
	return frac(b-a+1, b)
}

// SameLineThree 判断三点是否共线
func SameLineThree(p1, p2, p3 []int) bool {
	return p1[0]*(p2[1]-p3[1])-p2[0]*(p1[1]-p3[1])+p3[0]*(p1[1]-p2[1]) == 0
}

// SameLine 判断点集是否共线
func SameLine(points [][]int) bool {
	if len(points) < 3 {
		return true
	}
	a, b := points[0], points[1]
	for _, c := range points[2:] {
		if !SameLineThree(a, b, c) {
			return false
		}
	}
	return true
}

// ParseInt 转换10进制数到任意进制
func ParseInt(d int, base int) (num []int) {
	for {
		num = append(num, d%base)
		d = d / base
		if d == 0 {
			break
		}
	}
	reverseSlice(num)
	return
}
