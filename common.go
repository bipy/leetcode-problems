package leetcode

const mod int = 1e9 + 7

var dir4 = [4]struct{ x, y int }{{1, 0}, {0, -1}, {-1, 0}, {0, 1}}

func min(x, y int) int {
	if x < y {
		return x
	}
	return y
}

func max(x, y int) int {
	if x < y {
		return y
	}
	return x
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

func minIdx(nums []int) (idx int) {
	for i, v := range nums {
		if nums[idx] > v {
			idx = i
		}
	}
	return
}

func maxIdx(nums []int) (idx int) {
	for i, v := range nums {
		if nums[idx] < v {
			idx = i
		}
	}
	return
}

func minOf(nums ...int) int {
	ans := nums[0]
	for _, v := range nums {
		if ans > v {
			ans = v
		}
	}
	return ans
}

func maxOf(nums ...int) int {
	ans := nums[0]
	for _, v := range nums {
		if ans < v {
			ans = v
		}
	}
	return ans
}

func reverseSlice(arr []int) {
	for i := len(arr)/2 - 1; i >= 0; i-- {
		j := len(arr) - i - 1
		arr[i], arr[j] = arr[j], arr[i]
	}
}

func maxOfSlice(arr []int, less func(i, j int) bool) (idx int) {
	for i := 1; i < len(arr); i++ {
		if less(idx, i) {
			idx = i
		}
	}
	return
}

func minOfSlice(arr []int, less func(i, j int) bool) (idx int) {
	for i := 1; i < len(arr); i++ {
		if less(i, idx) {
			idx = i
		}
	}
	return
}

// RemoveDup 原地去重
func RemoveDup(arr []int) []int {
	set := map[int]struct{}{}
	for i := range arr {
		set[arr[i]] = struct{}{}
	}
	arr = arr[:0]
	for i := range set {
		arr = append(arr, i)
	}
	return arr
}

// Cond 三目运算
func Cond(cond bool, x, y interface{}) interface{} {
	if cond {
		return x
	}
	return y
}

// Zip 组合长度相同数组
func Zip(a ...[]int) [][]int {
	rt := make([][]int, len(a[0]))
	for i := range a[0] {
		for j := range a {
			rt[i] = append(rt[i], a[j][i])
		}
	}
	return rt
}

// ZipIdx 组合下标
func ZipIdx(a []int) []struct{ k, v int } {
	rt := make([]struct{ k, v int }, len(a))
	for i := range a {
		rt[i] = struct{ k, v int }{k: i, v: a[i]}
	}
	return rt
}

// Nums 生成连续数字数组 [begin, end)
func Nums(begin, end int) []int {
	n := end - begin
	rt := make([]int, n)
	for i := 0; i < n; i++ {
		rt[i] = begin + i
	}
	return rt
}

// Intersection 取交集
func Intersection(a, b []int) (rt []int) {
	if len(a) > len(b) {
		a, b = b, a
	}
	set := map[int]struct{}{}
	for i := range a {
		set[a[i]] = struct{}{}
	}
	for i := range b {
		if _, ok := set[b[i]]; ok {
			rt = append(rt, b[i])
		}
	}
	return
}

// XORSet 取异或集
func XORSet(a, b []int) (rt []int) {
	setA, setB := map[int]struct{}{}, map[int]struct{}{}
	for i := range a {
		setA[a[i]] = struct{}{}
	}
	for i := range b {
		setB[b[i]] = struct{}{}
	}
	for k := range setA {
		if _, ok := setB[k]; !ok {
			rt = append(rt, k)
		}
	}
	return
}

// CompareIntSlice 比较整数数组
func CompareIntSlice(a, b []int) int {
	if len(a) > len(b) {
		a, b = b, a
	}
	for i := 0; i < len(a); i++ {
		if a[i] < b[i] {
			return -1
		}
		if a[i] > b[i] {
			return 1
		}
	}
	if len(a) == len(b) {
		return 0
	}
	return -1
}

// CompareByteSlice 比较字节数组
func CompareByteSlice(a, b []byte) int {
	if len(a) > len(b) {
		a, b = b, a
	}
	for i := 0; i < len(a); i++ {
		if a[i] < b[i] {
			return -1
		}
		if a[i] > b[i] {
			return 1
		}
	}
	if len(a) == len(b) {
		return 0
	}
	return -1
}

func Filter[T any](arr []T, f func(i T) bool) []T {
	k := 0
	for i := range arr {
		if f(arr[i]) {
			arr[k] = arr[i]
			k++
		}
	}
	return arr[:k]
}

func FilterInt(arr []int, f func(i int) bool) []int {
	k := 0
	for i := range arr {
		if f(arr[i]) {
			arr[k] = arr[i]
			k++
		}
	}
	return arr[:k]
}

func FilterString(arr []string, f func(i string) bool) []string {
	k := 0
	for i := range arr {
		if f(arr[i]) {
			arr[k] = arr[i]
			k++
		}
	}
	return arr[:k]
}

func MapItems(m map[int]int) [][]int {
	rt := make([][]int, 0, len(m))
	for k, v := range m {
		rt = append(rt, []int{k, v})
	}
	return rt
}

func MapKeys(m map[int]int) []int {
	rt := make([]int, 0, len(m))
	for k := range m {
		rt = append(rt, k)
	}
	return rt
}

func MapValues(m map[int]int) []int {
	set := make(map[int]struct{})
	for _, v := range m {
		set[v] = struct{}{}
	}
	rt := make([]int, 0, len(set))
	for k := range set {
		rt = append(rt, k)
	}
	return rt
}

func Reduce(arr []int, f func(a, b int) int) int {
	if len(arr) == 0 {
		return 0
	}
	ans := arr[0]
	for i := 1; i < len(arr); i++ {
		ans = f(ans, arr[i])
	}
	return ans
}

func All(arr []int, f func(int) bool) bool {
	for _, v := range arr {
		if !f(v) {
			return false
		}
	}
	return true
}

func Any(arr []int, f func(int) bool) bool {
	for _, v := range arr {
		if f(v) {
			return true
		}
	}
	return false
}

func CountInt(arr []int) map[int]int {
	m := map[int]int{}
	for _, v := range arr {
		m[v]++
	}
	return m
}

func CountChar(s string) [26]int {
	cnt := [26]int{}
	base := 'a'
	if len(s) > 0 && s[0] >= 'A' && s[0] <= 'Z' {
		base = 'A'
	}
	for _, c := range s {
		cnt[c-base]++
	}
	return cnt
}

func IntRepeat(x int, count int) []int {
	if count == 0 {
		return []int{}
	}
	arr := make([]int, count)
	arr[0] = x
	for i := 1; i < len(arr); i <<= 1 {
		copy(arr[i:], arr[:i])
	}
	return arr
}
