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
	n := len(arr)
	mid := n >> 1
	left, right := arr[:mid], arr[mid:]
	MergeSort(left)
	MergeSort(right)
	t := make([]int, 0, n)
	for len(left) > 0 && len(right) > 0 {
		if left[0] < right[0] {
			t = append(t, left[0])
			left = left[1:]
		} else {
			t = append(t, right[0])
			right = right[1:]
		}
	}
	if len(left) > 0 {
		copy(arr, append(t, left...))
	} else {
		copy(arr, append(t, right...))
	}
}

func QuickSort(arr []int) {
	if len(arr) > 1 {
		i := -1
		pivot := arr[len(arr)-1]
		for j := range arr {
			if arr[j] <= pivot {
				i++
				arr[i], arr[j] = arr[j], arr[i]
			}
		}
		QuickSort(arr[:i])
		QuickSort(arr[i+1:])
	}
}

func RandomizedQuickSort(arr []int) {
	if len(arr) > 1 {
		i := -1
		p := rand.Intn(len(arr))
		arr[p], arr[len(arr)-1] = arr[len(arr)-1], arr[p]
		pivot := arr[len(arr)-1]
		for j := range arr {
			if arr[j] <= pivot {
				i++
				arr[i], arr[j] = arr[j], arr[i]
			}
		}
		RandomizedQuickSort(arr[:i])
		RandomizedQuickSort(arr[i+1:])
	}
}

func HeapSort(arr []int) {
	// max heaps
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
