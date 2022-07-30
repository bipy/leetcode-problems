package leetcode

func RemoveLastOne(x uint) uint {
	return x & (x - 1)
}

func Bin32(x int) [31]byte {
	s := [31]byte{}
	for i := range s {
		s[i] = byte(x >> (30 - i) & 1)
	}
	return s
}
