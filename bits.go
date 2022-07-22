package leetcode

func RemoveLastOne(x uint) uint {
	return x & (x - 1)
}
