package leetcode

import (
	"fmt"
	"sort"
	"testing"
)

func Test_alienOrder(t *testing.T) {
	fmt.Println(alienOrder(StrToStrSlice("[\"wrt\",\"wrf\",\"er\",\"ett\",\"rftt\"]")))
}

func Test_calculateTax(t *testing.T) {
	fmt.Println(calculateTax(StrTo2DIntSlice("[[3,50],[7,10],[12,25]]"), 10))
}

func Test_minPathCost(t *testing.T) {
	defer Timer()
	fmt.Println(minPathCost(StrTo2DIntSlice("[[5,3],[4,0],[2,1]]"), StrTo2DIntSlice("[[9,8],[1,5],[10,12],[18,6],[2,4],[14,3]]")))
}

func Test_duplicateZeros(t *testing.T) {
	arr := StrToIntSlice("[8,4,5,0,0,0,0,7]")
	duplicateZeros(arr)
	fmt.Println(arr)
}

func Test_insert(t *testing.T) {
	L := CycleListDeserialize("[3,3,5]")
	insert(L, 0)
	L.Show()
}

func Test_longestSubsequence(t *testing.T) {
	defer Timer()
	fmt.Println(longestSubsequence("111100010000011101001110001111000000001011101111111110111000011111011000010101110100110110001111001001011001010011010000011111101001101000000101101001110110000111101011000101", 11713332))
}

func Test_findFrequentTreeSum(t *testing.T) {
	defer Timer()
	root := TreeDeserialize("[5,2,-3]")
	fmt.Println(findFrequentTreeSum(root))
}

func Test_findSubstring(t *testing.T) {
	defer Timer()
	rt := findSubstring("ababaab", StrToStrSlice("[\"ab\",\"ba\",\"ba\"]"))
	fmt.Println(rt)
}

func Test_largestValues(t *testing.T) {
	fmt.Println(largestValues(TreeDeserialize("[1,3,2,5,3,null,9]")))
}

func Test_countPairs(t *testing.T) {
	defer Timer()
	fmt.Println(countPairs(7, StrTo2DIntSlice("[[0,2],[0,5],[2,4],[1,6],[5,4]]")))
}

func Test_countHousePlacements(t *testing.T) {
	fmt.Println(countHousePlacements(3))
}

func Test_maximumsSplicedArray(t *testing.T) {
	fmt.Println(maximumsSplicedArray(StrToIntSlice("[60,60,60]"), StrToIntSlice("[10,90,10]")))
}

func Test_findLUSlength(t *testing.T) {
	defer Timer()
	fmt.Println(findLUSlength(StrToStrSlice("[\"aba\",\"cdc\",\"eae\"]")))
}

func Test_mincostTickets(t *testing.T) {
	fmt.Println(mincostTickets(StrToIntSlice("[1,4,6,7,8,20]"), StrToIntSlice("[7,2,15]")))
}

func Test_diffWaysToCompute(t *testing.T) {
	defer Timer()
	fmt.Println(diffWaysToCompute("2-1-1"))
}

func Test_fourSum(t *testing.T) {
	my := fourSum(StrToIntSlice("[-4,-3,-2,-1,0,0,1,2,3,4]"), 0)
	//my := StrTo2DIntSlice("[[-2,-1,0,3],[-2,-1,1,2],[-4,-3,3,4],[-4,0,1,3],[-3,-1,1,3],[-2,0,0,2],[-1,0,0,1],[-4,-1,1,4],[-3,0,0,3],[-3,0,1,2],[-3,-1,0,4],[-4,-2,2,4],[-4,0,0,4],[-3,-2,2,3]]")
	it := StrTo2DIntSlice("[[-4,-3,3,4],[-4,-2,2,4],[-4,-1,1,4],[-4,-1,2,3],[-4,0,0,4],[-4,0,1,3],[-3,-2,1,4],[-3,-2,2,3],[-3,-1,0,4],[-3,-1,1,3],[-3,0,0,3],[-3,0,1,2],[-2,-1,0,3],[-2,-1,1,2],[-2,0,0,2],[-1,0,0,1]]")
	sort.Slice(my, func(i, j int) bool {
		for k := 0; k < 4; k++ {
			if my[i][k] != my[j][k] {
				return my[i][k] < my[j][k]
			}
		}
		return false
	})
	sort.Slice(it, func(i, j int) bool {
		for k := 0; k < 4; k++ {
			if it[i][k] != it[j][k] {
				return it[i][k] < it[j][k]
			}
		}
		return false
	})
	fmt.Println(my)
	fmt.Println(it)
}

func Test_reverseKGroup(t *testing.T) {
	reverseKGroup(ListDeserialize("[1,2,3,4,5]"), 2).Show()
}

func Test_isIsomorphic(t *testing.T) {
	isIsomorphic("add", "egg")
}

func Test_lowestCommonAncestor(t *testing.T) {
	tree := TreeDeserialize("[3,1,4,null,2]")
	p := tree
	q := tree.Left.Right
	fmt.Println(lowestCommonAncestor(tree, p, q).Val)
}

func Test_nextGreaterElement(t *testing.T) {
	nextGreaterElement(230241)
}

func Test_peopleAwareOfSecret(t *testing.T) {
	fmt.Println(peopleAwareOfSecret(289, 7, 23))
}

func Test_minTransfers(t *testing.T) {
	defer Timer()
	fmt.Println(minTransfers(StrTo2DIntSlice("[[1,8,1],[1,0,21],[2,8,10],[3,9,20],[4,10,61],[5,11,61],[6,1,59],[7,0,60]]")))
}

func Test_surroundedRegions(t *testing.T) {
	a := StrTo2DByteSlice("[[\"O\",\"O\",\"O\"],[\"O\",\"O\",\"O\"],[\"O\",\"O\",\"O\"]]")
	surroundedRegions(a)
	fmt.Println(a)
}

func Test_findCircleNum(t *testing.T) {
	findCircleNum(StrTo2DIntSlice("[[1,1,0],[1,1,0],[0,0,1]]"))
}

func Test_zeroFilledSubarray(t *testing.T) {
	zeroFilledSubarray(StrToIntSlice("[1,3,0,0,2,0,0,4]"))
}

func Test_shortestSequence(t *testing.T) {
	shortestSequence(StrToIntSlice("[1,1,2,2]"), 2)
}

func Test_countExcellentPairs(t *testing.T) {
	defer Timer()
	countExcellentPairs(StrToIntSlice("[1,2,3,1]"), 3)
}

func Test_fractionAddition(t *testing.T) {
	fractionAddition("-1/2+1/2")
}

func Test_largestComponentSize(t *testing.T) {
	defer Timer()
	args := GetInput()
	largestComponentSize(StrToIntSlice(args[0]))
}

func Test_maxLevelSum(t *testing.T) {
	rt := maxLevelSum(TreeDeserialize("[1,7,0,7,-8,null,null]"))
	fmt.Println(rt)
}

func Test_closestMeetingNode(t *testing.T) {
	closestMeetingNode(StrToIntSlice("[2,2,3,-1]"), 0, 1)
}

func Test_longestCycle(t *testing.T) {
	defer Timer()
	args := GetInput()
	rt := longestCycle(StrToIntSlice(args[0]))
	fmt.Println(rt)
}

func Test_minTimeToType(t *testing.T) {
	minTimeToType("bza")
}
