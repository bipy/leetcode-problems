package algo

func TopK(k int, arr []*Item, less func(i, j int) bool) []*Item {
	partition(k, arr, less)
	return arr[:k]
}
