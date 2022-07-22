package algo

import "sort"

func TopK(k int, arr []*Item, less func(i, j int) bool) []*Item {
	if len(arr) < k {
		return arr
	}
	sort.Slice(arr, less)
	return arr[:k]
}
