package leetcode

const mod = 1e9 + 7

var dir4 = []struct{ x, y int }{{1, 0}, {0, -1}, {-1, 0}, {0, 1}}
var dirRightDown = []struct{ x, y int }{{0, 1}, {1, 0}}

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

func minIdx(nums ...int) (idx int) {
	for i, v := range nums {
		if nums[idx] > v {
			idx = i
		}
	}
	return
}

func maxIdx(nums ...int) (idx int) {
	for i, v := range nums {
		if nums[idx] < v {
			idx = i
		}
	}
	return
}
