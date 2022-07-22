package algo

import "math/rand"

func BubbleSort(arr []int) {
	n := len(arr)
	flag := true
	for flag {
		flag = false
		for i := 1; i < n; i++ {
			if arr[i-1] > arr[i] {
				arr[i], arr[i-1] = arr[i-1], arr[i]
				flag = true
			}
		}
	}
}

func InsertionSort(arr []int) {
	n := len(arr)
	for i := 1; i < n; i++ {
		j := i - 1
		for j >= 0 && arr[i] < arr[j] {
			arr[j+1] = arr[j]
			j--
		}
		arr[j+1] = arr[i]
	}
}

func MergeSort(arr []int) {
	if len(arr) <= 1 {
		return
	}
	if len(arr) == 2 {
		if arr[0] > arr[1] {
			arr[0], arr[1] = arr[1], arr[0]
		}
		return
	}
	n := len(arr)
	mid := n >> 1
	MergeSort(arr[:mid])
	MergeSort(arr[mid:])
	p, q, i := 0, mid, 0
	t := make([]int, n)
	for p < mid && q < n {
		if arr[p] < arr[q] {
			t[i] = arr[p]
			p++
		} else {
			t[i] = arr[q]
			q++
		}
		i++
	}
	for p < mid {
		t[i] = arr[p]
		p++
		i++
	}
	for q < n {
		t[i] = arr[q]
		q++
		i++
	}
	for k := range t {
		arr[k] = t[k]
	}
}

func QuickSort(arr []int) {
	if len(arr) > 1 {
		l, r := 1, len(arr)-1
		for l <= r {
			if arr[l] <= arr[0] {
				l++
			} else if arr[r] > arr[0] {
				r--
			} else {
				arr[l], arr[r] = arr[r], arr[l]
				l++
				r--
			}
		}
		arr[0], arr[r] = arr[r], arr[0]
		QuickSort(arr[:r])
		QuickSort(arr[r+1:])
	}
}

func RandomizedQuickSort(arr []int) {
	if len(arr) > 1 {
		p := rand.Intn(len(arr))
		arr[0], arr[p] = arr[p], arr[0]
		l, r := 1, len(arr)-1
		for l <= r {
			if arr[l] <= arr[0] {
				l++
			} else if arr[r] > arr[0] {
				r--
			} else {
				arr[l], arr[r] = arr[r], arr[l]
				l++
				r--
			}
		}
		arr[0], arr[r] = arr[r], arr[0]
		RandomizedQuickSort(arr[:r])
		RandomizedQuickSort(arr[r+1:])
	}
}

func HeapSort(arr []int) {
	// max heap
	cmp := func(i, j int) bool {
		return arr[i] > arr[j]
	}
	buildHeap(arr, cmp)
	n := len(arr)
	for last := n - 1; last > 0; last-- {
		arr[0], arr[last] = arr[last], arr[0]
		heapify(arr[:last], 0, cmp)
	}
}

func buildHeap(arr []int, less func(i, j int) bool) {
	for i := len(arr) >> 1; i >= 0; i-- {
		heapify(arr, i, less)
	}
}

func heapify(arr []int, cur int, less func(i, j int) bool) {
	left, right := cur<<1, cur<<1+1
	target := cur
	if left < len(arr) && less(left, cur) {
		target = left
	}
	if right < len(arr) && less(right, target) {
		target = right
	}
	if target != cur {
		arr[target], arr[cur] = arr[cur], arr[target]
		heapify(arr, target, less)
	}
}

func SelectionSort(arr []int) {
	n := len(arr)
	for i := 0; i < n; i++ {
		target := i
		for j := i + 1; j < n; j++ {
			if arr[j] < arr[target] {
				target = j
			}
		}
		arr[i], arr[target] = arr[target], arr[i]
	}
}

func ShellSort(arr []int) {
	n := len(arr)
	h := 1
	for h < n/3 {
		h = 3*h + 1
	}
	for h >= 1 {
		for i := h; i < n; i++ {
			j := i - h
			for j >= 0 && arr[j] > arr[i] {
				arr[j+h] = arr[j]
				j -= h
			}
			arr[j+h] = arr[i]
		}
		h /= 3
	}
}
