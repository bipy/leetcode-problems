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

func reverseSlice(arr []any) {
	for i := len(arr)/2 - 1; i >= 0; i-- {
		j := len(arr) - i - 1
		arr[i], arr[j] = arr[j], arr[i]
	}
}
