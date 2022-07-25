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

func removeDup(arr []int) []int {
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
