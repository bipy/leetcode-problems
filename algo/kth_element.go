package algo

import (
	"math/rand"
	"time"
)

func partition(k int, arr []*Item, less func(i, j int) bool) {
	rand.Seed(time.Now().UnixNano())
	rand.Shuffle(len(arr), func(i, j int) { arr[i], arr[j] = arr[j], arr[i] })
	for l, r := 0, len(arr)-1; l < r; {
		// 切分元素 arr[l]
		i, j := l, r+1
		for {
			for i++; i < r && less(i, l); i++ {
			}
			for j--; j > l && less(l, j); j-- {
			}
			if i >= j {
				break
			}
			arr[i], arr[j] = arr[j], arr[i]
		}
		arr[l], arr[j] = arr[j], arr[l]
		if j == k {
			break
		} else if j < k {
			l = j + 1
		} else {
			r = j - 1
		}
	}
}

// KthElement 求第 k 大
func KthElement(k int, arr []*Item, less func(i, j int) bool) *Item {
	partition(k, arr, less)
	return arr[k] //  arr[:k+1]  arr[k:]
}
