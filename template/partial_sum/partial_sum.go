package partial_sum

type PartialSum struct {
	data []int
}

func InitPartialSum(arr []int) *PartialSum {
	n := len(arr)
	data := make([]int, n)
	data[0] = arr[0]
	for i := 1; i < n; i++ {
		data[i] = data[i-1] + arr[i]
	}
	return &PartialSum{data}
}

// Query sum[left, right] 闭区间
func (ps PartialSum) Query(left, right int) int {
	if left == 0 {
		return ps.data[right]
	}
	return ps.data[right] - ps.data[left-1]
}

// QueryAll sum of all
func (ps PartialSum) QueryAll() int {
	return ps.data[len(ps.data)-1]
}
