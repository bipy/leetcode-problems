package leetcode

import (
	"fmt"
	"sort"
	"strconv"
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

func Test_taskSchedulerII(t *testing.T) {
	taskSchedulerII(StrToIntSlice("[1,2,1,2,3,1]"), 3)
}

func Test_countBadPairs(t *testing.T) {
	rt := countBadPairs(StrToIntSlice("[4]"))
	fmt.Println(rt)
}

func Test_minimumReplacement(t *testing.T) {
	minimumReplacement(StrToIntSlice("[7,6,15,6,11,14,10]"))
}

func Test_longestIdealString(t *testing.T) {
	longestIdealString("acfgbd", 2)
}

func Test_validPartition(t *testing.T) {
	fmt.Println(validPartition(StrToIntSlice("[993335,993336,993337,993338,993339,993340,993341]")))
}

func Test_exclusiveTime(t *testing.T) {
	exclusiveTime(2, StrToStrSlice("[\"0:start:0\",\"1:start:2\",\"1:end:5\",\"0:end:6\"]"))
}

func Test_minSwaps1(t *testing.T) {
	fmt.Println(minSwaps1(StrTo2DIntSlice("[[0,0],[0,1]]")))
}

func Test_maxSum(t *testing.T) {
	maxSum(StrToIntSlice("[2,4,5,8,10]"), StrToIntSlice("[4,6,8,9]"))
}

func Test_splitString(t *testing.T) {
	fmt.Println(splitString("050043"))
}

func Test_checkPalindromeFormation(t *testing.T) {
	fmt.Println(checkPalindromeFormation("pvhmupgqeltozftlmfjjde", "yjgpzbezspnnpszebzmhvp"))
}

func Test_minInsertions(t *testing.T) {
	fmt.Println(minInsertions("(()))(()))()())))"))
}

func Test_smallestNumber(t *testing.T) {
	defer Timer()
	fmt.Println(smallestNumber("IIIDDD"))
}

func Test_maxEqualFreq(t *testing.T) {
	fmt.Println(maxEqualFreq(StrToIntSlice("[1,2]")))
}

func Test_merge(t *testing.T) {
	fmt.Println(merge(StrTo2DIntSlice("[[1,4],[2,3]]")))
}

func Test_shiftingLetters(t *testing.T) {
	fmt.Println(shiftingLetters("dztz", StrTo2DIntSlice("[[0,0,0],[1,1,1]]")))
}

func Test_maximumSegmentSum(t *testing.T) {
	defer Timer()
	args := GetInput()
	fmt.Println(maximumSegmentSum(StrToIntSlice(args[0]), StrToIntSlice(args[1])))
}

func Test_kSum(t *testing.T) {
	fmt.Println(kSum(StrToIntSlice("[-1,1]"), 1))
}

func Test_findLongestChain(t *testing.T) {
	defer Timer()
	args := GetInput()
	fmt.Println(findLongestChain(StrTo2DIntSlice(args[0])))
}

func Test_isStrictlyPalindromic(t *testing.T) {
	fmt.Println(isStrictlyPalindromic(9))
}

func Test_maximumRobots(t *testing.T) {
	fmt.Println(maximumRobots(StrToIntSlice("[11,12,74,67,37,87,42,34,18,90,36,28,34,20]"), StrToIntSlice("[18,98,2,84,7,57,54,65,59,91,7,23,94,20]"), 937))
}

func Test_reorderSpaces(t *testing.T) {
	reorderSpaces("aa ss    dd ")
}

func Test_lengthOfLIS(t *testing.T) {
	defer Timer()
	args := GetInput()
	n, _ := strconv.Atoi(args[1])
	fmt.Println(lengthOfLIS(StrToIntSlice(args[0]), n))
}

func Test_countPairs1(t *testing.T) {
	fmt.Println(countPairs1(StrToIntSlice(
		"[149,107,1,63,0,1,6867,1325,5611,2581,39,89,46,18,12,20,22,234]",
	)))
}

func Test_trimMean(t *testing.T) {
	trimMean(StrToIntSlice("[6,2,7,5,1,2,0,3,10,2,5,0,5,5,0,8,7,6,8,0]"))
}

func Test_smallestSubarrays(t *testing.T) {
	fmt.Println(smallestSubarrays(StrToIntSlice("[1,0]")))
}

func Test_minimumMoney(t *testing.T) {
	fmt.Println(minimumMoney(StrTo2DIntSlice("")))
}

func Test_reverseOddLevels(t *testing.T) {
	reverseOddLevels(TreeDeserialize("[2,3,5,8,13,21,34]"))
}

func Test_reformatNumber(t *testing.T) {
	reformatNumber("1-23-45 67")
}

func Test_threeEqualParts(t *testing.T) {
	threeEqualParts([]int{1, 1, 0, 1, 1, 0, 1, 1})
}

func Test_orangesRotting(t *testing.T) {
	defer Timer()
	fmt.Println(orangesRotting(StrTo2DIntSlice("[[2,1,1],[1,1,0],[0,1,1]]")))
}
