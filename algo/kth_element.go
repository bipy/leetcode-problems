package algo

import "sort"

func KthElement(k int, arr []*Item, less func(i, j int) bool) *Item {
	if len(arr) < k {
		return nil
	}
	sort.Slice(arr, less)
	return arr[k]
}
