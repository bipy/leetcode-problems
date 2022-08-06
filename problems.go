package leetcode

import (
	"bytes"
	"container/heap"
	"fmt"
	"github.com/emirpasic/gods/trees/redblacktree"
	"leetcode/template/heaps"
	"leetcode/template/union_find"
	"math"
	"math/bits"
	"math/rand"
	"net"
	"sort"
	"strconv"
	"strings"
	"time"
	"unicode"
)

func mostCommonWord(paragraph string, banned []string) string {
	paragraph += "$"
	bannedMap := map[string]bool{}
	for _, s := range banned {
		bannedMap[s] = true
	}
	cntMap := map[string]int{}
	pre := 0
	for i := 0; i < len(paragraph); i++ {
		if !unicode.IsLetter(rune(paragraph[i])) {
			cur := strings.ToLower(paragraph[pre:i])
			if !bannedMap[cur] {
				cntMap[cur]++
			}
			for i < len(paragraph) && !unicode.IsLetter(rune(paragraph[i])) {
				i++
			}
			pre = i
		}
	}
	var rt string
	max := 0
	for k, v := range cntMap {
		if max < v {
			max = v
			rt = k
		}
	}

	return rt
}

func findRepeatNumber(nums []int) int {
	m := map[int]bool{}
	for _, v := range nums {
		if _, ok := m[v]; ok {
			return v
		}
		m[v] = true
	}
	return 0
}

func reversePrint(head *ListNode) []int {
	if head == nil {
		return []int{}
	}
	rt := reversePrint(head.Next)
	rt = append(rt, head.Val)
	return rt
}

func minArray(numbers []int) int {
	if len(numbers) == 1 {
		return numbers[0]
	}
	if numbers[0] < numbers[len(numbers)-1] {
		return numbers[0]
	}
	for i := 1; i < len(numbers); i++ {
		if numbers[i] < numbers[i-1] {
			return numbers[i]
		}
	}
	return numbers[0]
}

func printNumbers(n int) []int {
	rt := make([]int, int(math.Pow10(n)-1))
	for i := 0; i < len(rt); i++ {
		rt[i] = i + 1
	}
	return rt
}

func exchange(nums []int) []int {
	l, r := 0, len(nums)
	for l < r {
		if nums[l]%2 == 0 && nums[r]%2 == 1 {
			nums[l], nums[r] = nums[r], nums[l]
		}
		if nums[l]%2 == 1 {
			l++
		}
		if nums[r]%2 == 0 {
			r--
		}
	}
	return nums
}

func reverseList(head *ListNode) *ListNode {
	if head == nil || head.Next == nil {
		return head
	}
	rt := reverseList(head.Next)
	head.Next.Next = head
	head.Next = nil
	return rt
}

func mirrorTree(root *TreeNode) *TreeNode {
	if root == nil {
		return nil
	}
	mirrorTree(root.Left)
	mirrorTree(root.Right)
	root.Left, root.Right = root.Right, root.Left
	return root
}

func lexicalOrder(n int) []int {
	data := make([]int, n)
	data[0] = 1
	for i := 1; i < n; i++ {
		pre := data[i-1]
		if pre*10 <= n {
			data[i] = pre * 10
		} else if pre+1 > n {
			t := pre
			for t >= 10 {
				t /= 10
			}
			data[i] = t + 1
		} else {
			data[i] = pre + 1
		}
	}
	return data
}

func spiralOrder(matrix [][]int) []int {
	var ans []int
	rows := len(matrix)
	if rows == 0 {
		return ans
	}
	directions := [][]int{{0, 1}, {1, 0}, {0, -1}, {-1, 0}}
	cols := len(matrix[0])
	visited := make([][]bool, rows)
	for i := 0; i < rows; i++ {
		visited[i] = make([]bool, cols)
		for j := 0; j < cols; j++ {
			visited[i][j] = false
		}
	}
	var dfs func(int, int, int)
	dfs = func(x int, y int, d int) {
		if !visited[x][y] {
			visited[x][y] = true
			ans = append(ans, matrix[x][y])
		}
		nx, ny := x+directions[d][0], y+directions[d][1]
		if nx >= rows || ny >= cols || nx < 0 || ny < 0 || visited[nx][ny] {
			if len(ans) == rows*cols {
				return
			}
			dfs(x, y, (d+1)%4)
		} else {
			dfs(nx, ny, d)
		}
	}
	dfs(0, 0, 0)
	return ans
}

func levelOrder(root *TreeNode) [][]int {
	if root == nil {
		return nil
	}
	var ans [][]int
	queue := []*TreeNode{root}
	for len(queue) > 0 {
		var level []int
		n := len(queue)
		for i := 0; i < n; i++ {
			cur := queue[0]
			level = append(level, cur.Val)
			if cur.Left != nil {
				queue = append(queue, cur.Left)
			}
			if cur.Right != nil {
				queue = append(queue, cur.Right)
			}
			queue = queue[1:]
		}
		ans = append(ans, level)
	}
	return ans
}

func toGoatLatin(sentence string) string {
	aeiou := []byte("aeiouAEIOU")
	vowels := map[byte]struct{}{}
	for _, v := range aeiou {
		vowels[v] = struct{}{}
	}
	ans := &strings.Builder{}
	words := strings.Split(sentence, " ")
	for i, v := range words {
		if i != 0 {
			ans.WriteByte(' ')
		}
		if _, ok := vowels[v[0]]; !ok {
			ans.WriteString(v[1:])
			ans.WriteByte(v[0])
		} else {
			ans.WriteString(v)
		}
		ans.WriteString("ma")
		ans.WriteString(strings.Repeat("a", i+1))
	}
	return ans.String()
}

func convertInteger(A int, B int) int {
	r := A ^ B
	ans := 0
	if r < 0 {
		r &= 0x7fffffff
		ans = 1
	}
	for r != 0 {
		r &= r - 1
		ans++
	}
	return ans
}

func buildTree(preorder []int, inorder []int) *TreeNode {
	if len(preorder) == 0 {
		return nil
	}
	root := &TreeNode{Val: preorder[0]}
	var idx int
	for idx = 0; idx < len(inorder); idx++ {
		if inorder[idx] == root.Val {
			break
		}
	}
	root.Left = buildTree(preorder[1:idx+1], inorder[:idx])
	root.Right = buildTree(preorder[idx+1:], inorder[idx+1:])
	return root
}

func isCompleteTree(root *TreeNode) bool {
	var queue []*TreeNode
	queue = append(queue, root)
	flag := false
	for len(queue) != 0 {
		n := len(queue)
		for i := 0; i < n; i++ {
			cur := queue[i]
			if cur.Left != nil {
				if flag {
					return false
				}
				queue = append(queue, cur.Left)
			} else {
				flag = true
			}
			if cur.Right != nil {
				if flag {
					return false
				}
				queue = append(queue, cur.Right)
			} else {
				flag = true
			}
		}
		queue = queue[n:]
	}
	return true
}

func uniquePaths(m int, n int) int {
	dp := make([]int, n)
	dp[0] = 1
	for i := 0; i < m; i++ {
		for j := 1; j < n; j++ {
			dp[j] += dp[j-1]
		}
	}
	return dp[n-1]
}

func minFallingPathSum(grid [][]int) int {
	minExcept := func(idx int, nums ...int) int {
		min := 0x7fffffff
		for i, v := range nums {
			if i == idx {
				continue
			}
			if v < min {
				min = v
			}
		}
		return min
	}
	n := len(grid)
	for i := 1; i < n; i++ {
		for j := 0; j < n; j++ {
			grid[i][j] += minExcept(j, grid[i]...)
		}
	}
	return minExcept(-1, grid[n-1]...)
}

func findPaths(m int, n int, maxMove int, startRow int, startColumn int) int {
	if maxMove == 0 {
		return 0
	}
	directions := [][]int{{1, 0}, {0, -1}, {-1, 0}, {0, 1}}
	dp := make([][][]int, m)
	for i, _ := range dp {
		dp[i] = make([][]int, n)
		for j, _ := range dp[i] {
			dp[i][j] = make([]int, maxMove+1)
			for k, _ := range dp[i][j] {
				dp[i][j][k] = -1
			}
		}
	}
	var dfs func(int, int, int) int
	dfs = func(curMove int, x int, y int) int {
		if x < 0 || y < 0 || x >= m || y >= n {
			if curMove == 0 {
				return 1
			}
			return 0
		}
		if dp[x][y][curMove] != -1 {
			return dp[x][y][curMove]
		}
		sum := 0
		for _, d := range directions {
			nx, ny := x+d[0], y+d[1]
			sum += dfs(curMove-1, nx, ny)
		}
		dp[x][y][curMove] = sum
		return sum
	}
	return dfs(maxMove, startRow, startColumn)
}

func countPrefixes(words []string, s string) int {
	ans := 0
	for _, w := range words {
		n := len(w)
		if n > len(s) {
			continue
		}
		flag := true
		for i := 0; i < n; i++ {
			if w[i] != s[i] {
				flag = false
				break
			}
		}
		if flag {
			ans++
		}
	}
	return ans
}

func minimumAverageDifference(nums []int) int {
	n := len(nums)
	for i := 1; i < n; i++ {
		nums[i] += nums[i-1]
	}
	total := nums[n-1]
	ans := math.MaxInt32
	rt := 0
	for i := 0; i < n; i++ {
		var preAvg, postAvg int
		preAvg = nums[i] / (i + 1)
		if i == n-1 {
			postAvg = 0
		} else {
			postAvg = (total - nums[i]) / (n - i - 1)
		}
		cur := preAvg - postAvg
		if cur < 0 {
			cur = -cur
		}
		if ans > cur {
			ans = cur
			rt = i
		}
	}
	return rt
}

func countUnguarded(m int, n int, guards [][]int, walls [][]int) int {
	// 0 not touched
	// 1 watched north
	// 2 watched south
	// 3 watched east
	// 4 watched west
	// 100 wall
	// 110 guard
	ans := 0
	g := make([][]byte, m)
	for i := range g {
		g[i] = make([]byte, n)
	}
	for _, wall := range walls {
		g[wall[0]][wall[1]] = 100
	}
	for _, guard := range guards {
		g[guard[0]][guard[1]] = 110
	}
	for _, guard := range guards {
		x, y := guard[0], guard[1]
		// north 1
		for i := 1; i < m; i++ {
			nx := x - i
			if nx >= 0 {
				if g[nx][y] == 2 || g[nx][y] == 100 || g[nx][y] == 110 {
					break
				}
				if g[nx][y] == 0 {
					g[nx][y] = 1
					ans++
				}
			}
		}
		// south 2
		for i := 1; i < m; i++ {
			nx := x + i
			if nx < m {
				if g[nx][y] == 1 || g[nx][y] == 100 || g[nx][y] == 110 {
					break
				}
				if g[nx][y] == 0 {
					g[nx][y] = 2
					ans++
				}
			}
		}
		// east 3
		for i := 1; i < n; i++ {
			ny := y + i
			if ny < n {
				if g[x][ny] == 4 || g[x][ny] == 100 || g[x][ny] == 110 {
					break
				}
				if g[x][ny] == 0 {
					g[x][ny] = 3
					ans++
				}
			}
		}
		// west 4
		for i := 1; i < n; i++ {
			ny := y - i
			if ny >= 0 {
				if g[x][ny] == 3 || g[x][ny] == 100 || g[x][ny] == 110 {
					break
				}
				if g[x][ny] == 0 {
					g[x][ny] = 4
					ans++
				}
			}
		}
	}
	return m*n - ans - len(guards) - len(walls)
}

func threeSumClosest(nums []int, target int) int {
	n := len(nums)
	sort.Ints(nums)
	ans := math.MaxInt32
	rt := 0
	for i := 0; i < n-2; i++ {
		left, right := i+1, n-1
		for left < right {
			sum := nums[i] + nums[left] + nums[right]
			if ans > abs(sum-target) {
				ans = abs(sum - target)
				rt = sum
			}
			if sum > target {
				right--
			} else if sum < target {
				left++
			}
		}
	}
	return rt
}

func getAllElements(root1 *TreeNode, root2 *TreeNode) []int {
	var inorder func(*TreeNode, *[]int)
	inorder = func(cur *TreeNode, data *[]int) {
		if cur == nil {
			return
		}
		inorder(cur.Left, data)
		*data = append(*data, cur.Val)
		inorder(cur.Right, data)
	}
	var tree1, tree2, rt []int
	inorder(root1, &tree1)
	inorder(root2, &tree2)
	p1, p2 := 0, 0
	for p1 < len(tree1) && p2 < len(tree2) {
		if tree1[p1] < tree2[p2] {
			rt = append(rt, tree1[p1])
			p1++
		} else {
			rt = append(rt, tree2[p2])
			p2++
		}
	}
	if p1 == len(tree1) {
		rt = append(rt, tree2[p2:]...)
	} else {
		rt = append(rt, tree1[p1:]...)
	}
	return rt
}

func removeDigit(number string, digit byte) string {
	ans := ""
	arr := []byte(number)
	for i := range arr {
		if number[i] == digit {
			t := strings.Builder{}
			t.Write(arr[:i])
			t.Write(arr[i+1:])
			s := t.String()
			if ans == "" || strings.Compare(ans, s) == -1 {
				ans = s
			}
		}
	}
	return ans
}

func minimumCardPickup(cards []int) int {
	m := map[int]int{}
	ans := math.MaxInt32
	for i := 0; i < len(cards); i++ {
		if pre, ok := m[cards[i]]; ok {
			if ans > i-pre {
				ans = i - pre + 1
			}
		}
		m[cards[i]] = i
	}
	if ans == math.MaxInt32 {
		return -1
	}
	return ans
}

func countDistinct(nums []int, k int, p int) int {
	if p == 1 {
		// n - 1 + 1
		// n - 2 + 1
		// n - 3 + 1
		// n + (n - 1) + (n - 2) + ... + (n - (k - 1)) = k*n - (k-1)*k/2
		return k*len(nums) - (k-1)*k/2
	}
	var idx []int
	for i := range nums {
		if nums[i]%p == 0 {
			idx = append(idx, i)
		}
	}
	return 0

}

func appealSum(s string) int {
	cm := &CntMap{}
	n := len(s)
	ans := n
	cm.Add(s[0])
	for i := 2; i <= n; i++ {
		if i%2 == 0 {
			cm.Add(s[i-1])
			ans += cm.Len
			for j := i; j < n; j++ {
				cm.Add(s[j])
				cm.Remove(s[j-i])
				ans += cm.Len
			}
		} else {
			cm.Add(s[n-i])
			ans += cm.Len
			for j := n - i - 1; j >= 0; j-- {
				cm.Add(s[j])
				cm.Remove(s[j+i])
				ans += cm.Len
			}
		}
	}
	return ans
}

func reorderLogFiles(logs []string) []string {
	type node struct {
		idx     int
		isDigit bool
		sep     int
	}
	arr := make([]*node, 0, len(logs))
	for i, v := range logs {
		bs := strings.IndexByte(v, ' ')
		fc := v[bs+1]
		arr = append(arr, &node{
			idx:     i,
			isDigit: fc <= '9' && fc >= '0',
			sep:     bs,
		})
	}
	sort.SliceStable(arr, func(i, j int) bool {
		if arr[i].isDigit && arr[j].isDigit {
			return false
		}
		if !arr[i].isDigit && !arr[j].isDigit {
			if r := strings.Compare(logs[arr[i].idx][arr[i].sep:], logs[arr[j].idx][arr[j].sep:]); r == 0 {
				return strings.Compare(logs[arr[i].idx][:arr[i].sep], logs[arr[j].idx][:arr[j].sep]) == -1
			} else {
				return r == -1
			}
		}
		return arr[j].isDigit
	})
	rt := make([]string, 0, len(logs))
	for _, v := range arr {
		rt = append(rt, logs[v.idx])
	}
	return rt
}

func intersection(nums [][]int) []int {
	m := map[int]int{}
	for _, arr := range nums {
		for _, v := range arr {
			m[v]++
		}
	}
	var rt []int
	for k, v := range m {
		if v == len(nums) {
			rt = append(rt, k)
		}
	}
	sort.Ints(rt)
	return rt
}

func countLatticePoints(circles [][]int) int {
	const MAXL = 205
	g := make([][]bool, MAXL)
	for i := range g {
		g[i] = make([]bool, MAXL)
	}
	for _, circle := range circles {
		x, y, r := circle[0], circle[1], circle[2]
		r2 := r * r
		for i := x - r; i <= x+r; i++ {
			for j := y - r; j <= y+r; j++ {
				if getDistSq(i-x, j-y) <= r2 {
					g[i][j] = true
				}
			}
		}
	}
	ans := 0
	for i := range g {
		for j := range g[i] {
			if g[i][j] {
				ans++
			}
		}
	}
	return ans
}

func countRectangles(rectangles [][]int, points [][]int) []int {
	recs := make([][]int, 101)
	for _, r := range rectangles {
		recs[r[1]] = append(recs[r[1]], r[0])
	}
	for i := range recs {
		sort.Ints(recs[i])
	}
	rt := make([]int, len(points))
	for i := range points {
		for j := points[i][1]; j < len(recs); j++ {
			rt[i] += len(recs[j]) - sort.SearchInts(recs[j], points[i][0])
		}
	}
	return rt
}

func minMutation(start string, end string, bank []string) int {
	isOK := func(a, b string) bool {
		cnt := 0
		for i := range a {
			if a[i] != b[i] {
				cnt++
			}
			if cnt > 1 {
				return false
			}
		}
		return cnt == 1
	}
	queue := []string{start}
	vis := map[string]bool{start: true}
	ans := 0
	for len(queue) != 0 {
		ans++
		n := len(queue)
		for i := 0; i < n; i++ {
			cur := queue[i]
			for _, v := range bank {
				if !vis[v] && isOK(cur, v) {
					if v == end {
						return ans
					}
					vis[v] = true
					queue = append(queue, v)
				}
			}
		}
		queue = queue[n:]
	}
	return -1
}

func largestGoodInteger(num string) string {
	ans := ""
	for i := 2; i < len(num); i++ {
		if num[i-2] == num[i-1] && num[i-1] == num[i] {
			if ans < num[i-2:i+1] {
				ans = num[i-2 : i+1]
			}
		}
	}
	return ans
}

func averageOfSubtree(root *TreeNode) int {
	ans := 0
	var dfs func(*TreeNode) (int, int)
	dfs = func(cur *TreeNode) (int, int) {
		if cur == nil {
			return 0, 0
		}
		lsum, lcnt := dfs(cur.Left)
		rsum, rcnt := dfs(cur.Right)
		sum, cnt := lsum+rsum+cur.Val, lcnt+rcnt+1
		if sum/cnt == cur.Val {
			ans++
		}
		return sum, cnt
	}
	dfs(root)
	return ans
}

func countTexts(pressedKeys string) int {
	pressedKeys += "$"
	dp3, dp4 := []int{0, 1, 2, 4}, []int{0, 1, 2, 4, 8}
	get3 := func(n int) int {
		for i := len(dp3); i < n+1; i++ {
			dp3 = append(dp3, dp3[i-1]+dp3[i-2]+dp3[i-3])
			dp3[i] %= mod
		}
		return dp3[n]
	}
	get4 := func(n int) int {
		for i := len(dp4); i < n+1; i++ {
			dp4 = append(dp4, dp4[i-1]+dp4[i-2]+dp4[i-3]+dp4[i-4])
			dp4[i] %= mod
		}
		return dp4[n]
	}
	pre := 0
	ans := 0
	for i := range pressedKeys {
		if pressedKeys[i] != pressedKeys[pre] {
			if pressedKeys[pre] == '7' || pressedKeys[pre] == '9' {
				ans *= get4(i - pre)
			} else {
				ans *= get3(i - pre)
			}
			ans %= mod
			pre = i
		}
	}
	return ans
}

func hasValidPath(grid [][]byte) bool {
	m, n := len(grid), len(grid[0])
	if grid[m-1][n-1] == '(' {
		return false
	}
	dir := [2]struct{ x, y int }{{1, 0}, {0, 1}}
	dp := make([][]bool, m)
	for i := range dp {
		dp[i] = make([]bool, n)
	}
	ans := false
	var path []byte
	var dfs func(int, int)
	dfs = func(x int, y int) {
		if x >= m || y >= n || ans {
			return
		}
		if dp[x][y] {
			return
		}
		if grid[x][y] == '(' {
			if len(path) == 0 {
				dp[x][y] = true
			}
			path = append(path, '(')
			for _, d := range dir {
				nx, ny := x+d.x, y+d.y
				dfs(nx, ny)
			}
			path = path[:len(path)-1]
		} else {
			if len(path) == 0 {
				return
			}
			path = path[:len(path)-1]
			if x == m-1 && y == n-1 && len(path) == 0 {
				ans = true
			}
			for _, d := range dir {
				nx, ny := x+d.x, y+d.y
				dfs(nx, ny)
			}
			path = append(path, '(')
		}
	}
	return ans
}

func digitSum(s string, k int) string {
	for len(s) > k {
		n := len(s) / k
		if len(s)%k != 0 {
			n++
		}
		min := func(x, y int) int {
			if x < y {
				return x
			}
			return y
		}
		sb := strings.Builder{}
		for i := 0; i < n; i++ {
			sum := 0
			cur := s[i*k : min((i+1)*k, len(s))]
			for j := range cur {
				sum += int(cur[j] - '0')
			}
			sb.WriteString(strconv.Itoa(sum))
		}
		s = sb.String()
	}
	return s
}

func minimumRounds(tasks []int) int {
	cnt := map[int]int{}
	for _, v := range tasks {
		cnt[v]++
	}
	ans := 0
	for _, v := range cnt {
		n := v / 3
		if v%3 == 0 {
			ans += n
		} else {
			flag := false
			for i := n; i >= 0; i-- {
				if (v-i*3)%2 == 0 {
					flag = true
					ans += i + (v-i*3)/2
					break
				}
			}
			if !flag {
				return -1
			}
		}
	}
	return ans
}

func maxTrailingZeros(grid [][]int) int {
	m, n := len(grid), len(grid[0])
	directions := [][]int{{1, 0}, {0, 1}, {-1, 0}, {0, -1}}
	ans := 0
	var dfs func(int, int, int, int, bool)
	dfs = func(x int, y int, prod int, dirIdx int, flag bool) {
		cur := 0
		for i := 10; i <= prod; i *= 10 {
			if prod%i == 0 {
				cur++
			}
		}
		if cur > ans {
			ans = cur
		}
		if flag {
			nx, ny := x+directions[dirIdx][0], y+directions[dirIdx][1]
			if nx >= 0 && nx < m && ny >= 0 && ny < n {
				dfs(nx, ny, prod*grid[nx][ny], dirIdx, true)
			}
		} else {
			for idx, d := range directions {
				if idx-dirIdx == 2 || dirIdx-idx == 2 {
					continue
				}
				nx, ny := x+d[0], y+d[1]
				if nx >= 0 && nx < m && ny >= 0 && ny < n {
					if idx == dirIdx {
						dfs(nx, ny, prod*grid[nx][ny], idx, false)
					} else {
						dfs(nx, ny, prod*grid[nx][ny], idx, true)
					}
				}
			}
		}
	}
	for i := range grid {
		for j := range grid[i] {
			for d := range directions {
				dfs(i, j, grid[i][j], d, false)
			}
		}
	}
	return ans
}

func divisorSubstrings(num int, k int) int {
	snum := strconv.Itoa(num)
	ans := 0
	for i := k; i <= len(snum); i++ {
		cur, _ := strconv.Atoi(snum[i-k : i])
		if cur != 0 && num%cur == 0 {
			ans++
		}
	}
	return ans
}

func waysToSplitArray(nums []int) int {
	ans := 0
	n := len(nums)
	for i := 1; i < n; i++ {
		nums[i] += nums[i-1]
	}
	for i := 0; i < n-1; i++ {
		pre := nums[i]
		post := nums[n-1] - nums[i]
		if pre >= post {
			ans++
		}
	}
	return ans
}

func maximumWhiteTiles(tiles [][]int, carpetLen int) int {
	sort.Slice(tiles, func(i, j int) bool {
		return tiles[i][0] < tiles[j][0]
	})
	max := func(x, y int) int {
		if x < y {
			return y
		}
		return x
	}
	n := len(tiles)
	ans := 0
	sums := make([]int, n)
	sums[0] = tiles[0][1] - tiles[0][0] + 1
	for i := 1; i < n; i++ {
		sums[i] += sums[i-1] + tiles[i][1] - tiles[i][0] + 1
	}
	for i := 0; i < n; i++ {
		last := sort.Search(n, func(k int) bool {
			return tiles[k][1] >= tiles[i][1]-carpetLen
		})
		sum := tiles[last][1] - max(tiles[i][1]-carpetLen+1, tiles[last][0]) + 1 + sums[i] - sums[last]
		ans = max(ans, sum)
	}
	return ans
}

func largestVariance(s string) int {
	ans := 0
	maxf := func(x, y int) int {
		if x > y {
			return x
		}
		return y
	}
	for i := range s {
		cnt := [26]int{}
		vis := map[byte]struct{}{}
		max, min := s[i]-'a', s[i]-'a'
		cnt[s[i]-'a']++
		vis[s[i]-'a'] = struct{}{}
		for j := i + 1; j < len(s); j++ {
			cur := s[j] - 'a'
			vis[cur] = struct{}{}
			cnt[cur]++
			if cnt[cur] < cnt[min] {
				min = cur
			} else if cur == min {
				for k := range vis {
					if cnt[k] < cnt[min] {
						min = k
					}
				}
			}
			if cnt[cur] > cnt[max] {
				max = cur
			}
			ans = maxf(ans, cnt[max]-cnt[min])
		}
	}
	return ans
}

func largestTriangleArea(points [][]int) float64 {
	n := len(points)
	ans := 0.0
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			for k := j + 1; k < n; k++ {
				ans = math.Max(ans, math.Abs(float64(points[i][0]*(points[j][1]-points[k][1])-points[j][0]*(points[i][1]-points[k][1])+points[k][0]*(points[i][1]-points[j][1])))*0.5)
			}
		}
	}
	return ans
}

func isAlienSorted(words []string, order string) bool {
	orders := [26]int{}
	for i, c := range order {
		orders[c] = i
	}
	return sort.SliceIsSorted(words, func(i, j int) bool {
		for p := 0; p < len(words[i]) && p < len(words[j]); p++ {
			if orders[words[i][p]] == orders[words[j][p]] {
				continue
			}
			return orders[words[i][p]] < orders[words[j][p]]
		}
		return len(words[i]) < len(words[j])
	})
}

func removeAnagrams(words []string) []string {
	n := len(words)
	check := func(i, j int) bool {
		if len(words[i]) != len(words[j]) {
			return false
		}
		cnt := [26]int{}
		for _, c := range words[i] {
			cnt[c-'a']++
		}
		for _, c := range words[j] {
			cnt[c-'a']--
		}
		for k := range cnt {
			if cnt[k] != 0 {
				return false
			}
		}
		return true
	}
	pre := 0
	var ans []string
	ans = append(ans, words[0])
	for i := 1; i < n; i++ {
		if check(i, pre) {
			continue
		}
		pre = i
		ans = append(ans, words[i])
	}
	return ans
}

func maxConsecutive(bottom int, top int, special []int) int {
	sort.Ints(special)
	ans := special[0] - bottom
	for i := 1; i < len(special); i++ {
		ans = max(ans, special[i]-special[i-1]-1)
	}
	return max(ans, top-special[len(special)-1])
}

func largestCombination(candidates []int) int {
	cnt := [32]int{}
	for _, v := range candidates {
		for i := 0; v != 0; i++ {
			if v&1 == 1 {
				cnt[i]++
			}
			v >>= 1
		}
	}
	ans := 0
	for i := range cnt {
		if cnt[i] > ans {
			ans = cnt[i]
		}
	}
	return ans
}

func minMoves2(nums []int) int {
	sort.Ints(nums)
	target := nums[len(nums)>>1]
	ans := 0
	for _, v := range nums {
		ans += abs(v - target)
	}
	return ans
}

type CountIntervals struct {
	*redblacktree.Tree
	cnt int // 所有区间长度和
}

func ConstructorCountIntervals() CountIntervals {
	return CountIntervals{redblacktree.NewWithIntComparator(), 0}
}

func (t *CountIntervals) Add(left, right int) {
	//// 遍历所有被 [left,right] 覆盖到的区间（部分覆盖也算）
	//for node, _ := t.Ceiling(left); node != nil && node.Value.(int) <= right; node, _ = t.Ceiling(left) {
	//	l, r := node.Value.(int), node.Key.(int)
	//	if l < left {
	//		left = l
	//	}
	//	if r > right {
	//		right = r
	//	}
	//	t.cnt -= r - l + 1
	//	t.Remove(r)
	//}
	//t.cnt += right - left + 1
	//t.Put(right, left)
	for node, ok := t.Floor(right); ok && node.Key.(int) >= left; node, ok = t.Floor(right) {
		l, r := node.Key.(int), node.Value.(int)
		if l < left {
			left = l
		}
		if r > right {
			right = r
		}
		t.cnt -= r - l + 1
		t.Remove(l)
	}
	t.cnt += right - left + 1
	t.Put(left, right)
}

func (t *CountIntervals) Count() int { return t.cnt }

func findRightInterval(intervals [][]int) []int {
	n := len(intervals)
	idx := make([]int, n)
	for i := range idx {
		idx[i] = i
	}
	sort.Slice(idx, func(i, j int) bool {
		return intervals[idx[i]][0] < intervals[idx[j]][0]
	})
	ans := make([]int, n)
	for i := 0; i < n; i++ {
		target := intervals[idx[i]][1]
		p := sort.Search(n, func(k int) bool {
			return intervals[idx[k]][0] >= target
		})
		if p == n {
			ans[idx[i]] = -1
		} else {
			ans[idx[i]] = idx[p]
		}
	}
	return ans
}

func repeatedNTimes(nums []int) int {
	n := len(nums)
	for {
		x, y := rand.Intn(n), rand.Intn(n)
		if x != y && nums[x] == nums[y] {
			return nums[x]
		}
	}
}

func percentageLetter(s string, letter byte) int {
	cnt := 0
	for i := range s {
		if s[i] == letter {
			cnt++
		}
	}
	return cnt * 100 / len(s)
}

func maximumBags(capacity []int, rocks []int, additionalRocks int) int {
	n := len(capacity)
	cnt := make([]int, n)
	ans := 0
	for i := range cnt {
		cnt[i] = capacity[i] - rocks[i]
	}
	sort.Ints(cnt)
	for i := range cnt {
		additionalRocks -= cnt[i]
		if additionalRocks >= 0 {
			ans++
		} else {
			break
		}
	}
	return ans
}

func minimumLines(stockPrices [][]int) int {
	sort.Slice(stockPrices, func(i, j int) bool {
		return stockPrices[i][0] < stockPrices[j][0]
	})
	stockPrices = append(stockPrices, []int{stockPrices[len(stockPrices)-1][0] + 1, math.MaxInt})
	n := len(stockPrices)
	ans := 0
	for i := 2; i < n; i++ {
		x1, y1 := stockPrices[i-2][0], stockPrices[i-2][1]
		x2, y2 := stockPrices[i-1][0], stockPrices[i-1][1]
		x3, y3 := stockPrices[i][0], stockPrices[i][1]
		if x1*(y2-y3)-x2*(y1-y3)+x3*(y1-y2) != 0 {
			ans++
		}
	}
	return ans
}

func isUnivalTree(root *TreeNode) bool {
	target := root.Val
	var dfs func(*TreeNode) bool
	dfs = func(cur *TreeNode) bool {
		if cur == nil {
			return true
		}
		if cur.Val != target {
			return false
		}
		return dfs(cur.Left) && dfs(cur.Right)
	}
	return dfs(root)
}

func findSubstringInWraproundString(p string) int {
	n := len(p)
	dp := [26]int{}
	k := 0
	for i := 1; i < n; i++ {
		if p[i]-'a' == (p[i-1]-'a'+1)%26 {
			k++
		} else {
			k = 1
		}
		dp[p[i]-'a'] = max(dp[p[i]-'a'], k)
	}
	ans := 0
	for _, v := range dp {
		ans += v
	}
	return ans
}

func removeOuterParentheses(s string) string {
	cnt := 0
	sb := strings.Builder{}
	sb.Grow(len(s))
	for i := range s {
		if s[i] == '(' {
			if cnt != 0 {
				sb.WriteByte('(')
			}
			cnt++
		} else {
			cnt--
			if cnt != 0 {
				sb.WriteByte(')')
			}
		}
	}
	return sb.String()
}

func digitCount(num string) bool {
	cnt := [10]int{}
	for i := range num {
		cnt[num[i]-'0']++
	}
	for i := 0; i < len(num); i++ {
		if cnt[i] != int(num[i]-'0') {
			return false
		}
	}
	return true
}

func largestWordCount(messages []string, senders []string) string {
	cnt := map[string]int{}
	n := len(messages)
	for i := 0; i < n; i++ {
		cnt[senders[i]] += len(strings.Split(messages[i], " "))
	}
	maxv := 0
	for _, v := range cnt {
		if maxv < v {
			maxv = v
		}
	}
	ans := ""
	for k, v := range cnt {
		if maxv == v {
			if ans < k {
				ans = k
			}
		}
	}
	return ans
}

func maximumImportance(n int, roads [][]int) int64 {
	list := make([]int, n)
	for _, road := range roads {
		list[road[0]]++
		list[road[1]]++
	}
	sort.Slice(list, func(i, j int) bool {
		return list[i] > list[j]
	})
	var ans int64 = 0
	w := n
	for _, v := range list {
		ans += int64(w * v)
		w--
	}
	return ans
}

func validIPAddress(queryIP string) string {
	ip := net.ParseIP(queryIP)
	if ip == nil {
		return "Neither"
	}
	if ip.To4() == nil {
		items := strings.Split(queryIP, ":")
		for _, s := range items {
			if l := len(s); l < 1 || l > 4 {
				return "Neither"
			}
		}
		return "IPv6"
	}
	items := strings.Split(queryIP, ".")
	for _, s := range items {
		if len(s) > 1 && s[0] == '0' {
			return "Neither"
		}
	}
	return "IPv4"
}

func rearrangeCharacters(s string, target string) int {
	cnt, targetCnt := [26]int{}, [26]int{}
	for i := range s {
		cnt[s[i]-'a']++
	}
	for i := range target {
		targetCnt[target[i]-'a']++
	}
	ans := 0x7fffffff
	for i := range cnt {
		if targetCnt[i] != 0 && ans > cnt[i]/targetCnt[i] {
			ans = cnt[i] / targetCnt[i]
		}
	}
	return ans
}

func discountPrices(sentence string, discount int) string {
	words := strings.Split(sentence, " ")
	disc := float64(100-discount) / 100.0
	for i := range words {
		if len(words[i]) > 1 && words[i][0] == '$' {
			if input, err := strconv.Atoi(words[i][1:]); err == nil {
				words[i] = fmt.Sprintf("$%.2f", float64(input)*disc)
			}
		}
	}
	return strings.Join(words, " ")
}

func minimumObstacles(grid [][]int) int {
	m, n := len(grid), len(grid[0])
	dp := make([][]int, m)
	for i := range dp {
		dp[i] = make([]int, n)
		for j := range dp[i] {
			dp[i][j] = 0x7fffffff
		}
	}
	type node struct {
		x, y, val int
	}
	queue := make([]node, 0, m*n)
	queue = append(queue, node{
		x:   0,
		y:   0,
		val: 0,
	})
	for len(queue) > 0 {
		cur := queue[0]
		if cur.val <= dp[cur.x][cur.y] {
			for _, d := range dir4 {
				nx, ny := cur.x+d.x, cur.y+d.y
				if nx >= 0 && nx < m && ny >= 0 && ny < n {
					nval := cur.val + grid[nx][ny]
					if dp[nx][ny] > nval {
						dp[nx][ny] = nval
						queue = append(queue, node{
							x:   nx,
							y:   ny,
							val: nval,
						})
					}
				}
			}
		}
		queue = queue[1:]
	}
	return dp[m-1][n-1]
}

func sumRootToLeaf(root *TreeNode) (ans int) {
	var dfs func(*TreeNode, int)
	dfs = func(cur *TreeNode, val int) {
		val += cur.Val
		if cur.Left == nil && cur.Right == nil {
			ans += val
			return
		}
		if cur.Left != nil {
			dfs(cur.Left, val<<1)
		}
		if cur.Right != nil {
			dfs(cur.Right, val<<1)
		}
	}
	dfs(root, root.Val<<1)
	return
}

func alienOrder(words []string) string {
	g := [26][26]int{}
	cnt := map[int]int{}
	n := len(words)
	for _, w := range words {
		for _, c := range w {
			cnt[int(c-'a')] = 0
		}
	}
	for i := 1; i < n; i++ {
		flag := false
		k := min(len(words[i-1]), len(words[i]))
		for j := 0; j < k; j++ {
			if c1, c2 := words[i-1][j], words[i][j]; c1 != c2 {
				g[c1-'a'][c2-'a'] = 1
				flag = true
				break
			}
		}
		if !flag && len(words[i-1]) > len(words[i]) {
			return ""
		}
	}
	for i := 0; i < 26; i++ {
		for j := 0; j < 26; j++ {
			if g[i][j] == 1 {
				cnt[j]++
			}
		}
	}
	sb := strings.Builder{}
	for {
		flag := false
		for k, v := range cnt {
			if v == 0 {
				flag = true
				sb.WriteByte('a' + byte(k))
				for j := 0; j < 26; j++ {
					if g[k][j] == 1 {
						cnt[j]--
					}
				}
				delete(cnt, k)
				break
			}
		}
		if !flag {
			break
		}
	}
	if len(cnt) > 0 {
		return ""
	}
	return sb.String()
}

func deleteNode(root *TreeNode, key int) *TreeNode {
	R := &TreeNode{Right: root}
	pre, p := R, root
	for p != nil && p.Val != key {
		pre = p
		if p.Val > key {
			p = p.Left
		} else {
			p = p.Right
		}
	}
	if pre.Left == p {
		smallest := p.Right
		if smallest == nil {
			pre.Left = p.Left
		} else {
			pre.Left = p.Right
			for smallest.Left != nil {
				smallest = smallest.Left
			}
			smallest.Left = p.Left
		}
	} else {
		largest := p.Left
		if largest == nil {
			pre.Right = p.Right
		} else {
			pre.Right = p.Left
			for largest.Right != nil {
				largest = largest.Right
			}
			largest.Right = p.Right
		}
	}
	return R.Right
}

func sumOfThree(num int64) []int64 {
	if num%3 == 0 {
		return []int64{num/3 - 1, num / 3, num/3 + 1}
	}
	return []int64{}
}

func consecutiveNumbersSum(n int) int {
	ans := 1
	for i := 2; i < n; i++ {
		if t := n - i*(i-1)/2; t%i == 0 {
			if t < 0 {
				break
			}
			ans++
		}
	}
	return ans
}

func numUniqueEmails(emails []string) int {
	m := map[string]struct{}{}
	for _, email := range emails {
		sb := strings.Builder{}
		for i := range email {
			if email[i] == '@' || email[i] == '+' {
				for email[i] != '@' {
					i++
				}
				sb.WriteString(email[i:])
				break
			} else if email[i] != '.' {
				sb.WriteByte(email[i])
			}
		}
		m[sb.String()] = struct{}{}
	}
	return len(m)
}

func isBoomerang(points [][]int) bool {
	x1, y1 := points[0][0], points[0][1]
	x2, y2 := points[1][0], points[1][1]
	x3, y3 := points[2][0], points[2][1]
	return x1*(y2-y3)-x2*(y1-y3)+x3*(y1-y2) != 0
}

func minEatingSpeed(piles []int, h int) int {
	upper := 0
	for i := range piles {
		if piles[i] > upper {
			upper = piles[i]
		}
	}
	return 1 + sort.Search(upper, func(k int) bool {
		k++
		time := 0
		for i := range piles {
			time += piles[i] / k
			if piles[i]%k != 0 {
				time++
			}
		}
		return time <= h
	})
}

func minFlipsMonoIncr(s string) int {
	ans := len(s)
	zero, one := 0, 0
	for i := range s {
		if s[i] == '0' {
			zero++
		}
	}
	for i := range s {
		ans = min(ans, one+zero-(i-one))
		if s[i] == '1' {
			one++
		}
	}
	ans = min(ans, one+zero-(len(s)-one))
	return ans
}

func calculateTax(brackets [][]int, income int) float64 {
	if income < brackets[0][0] {
		return float64(income) * float64(brackets[0][1]) / 100
	}
	ans := float64(brackets[0][0]) * float64(brackets[0][1]) / 100
	for i := 1; i < len(brackets); i++ {
		if income >= brackets[i][0] {
			ans += float64(brackets[i][0]-brackets[i-1][0]) * float64(brackets[i][1]) / 100
		} else {
			ans += float64(income-brackets[i-1][0]) * float64(brackets[i][1]) / 100
			break
		}
	}
	return ans
}

func minPathCost(grid [][]int, moveCost [][]int) int {
	m, n := len(grid), len(grid[0])
	dp := make([][]int, m)
	for i := range dp {
		dp[i] = make([]int, n)
	}
	for i := 1; i < m; i++ {
		for j := 0; j < n; j++ {
			minVal := 0x7fffffff
			for k := 0; k < n; k++ {
				if t := dp[i-1][k] + moveCost[grid[i-1][k]][j] + grid[i-1][k]; minVal > t {
					minVal = t
				}
			}
			dp[i][j] = minVal
		}
	}
	for i := range dp[m-1] {
		dp[m-1][i] += grid[m-1][i]
	}
	ans := 0x7fffffff
	for _, v := range dp[m-1] {
		if ans > v {
			ans = v
		}
	}
	return ans
}

func distributeCookies(cookies []int, k int) int {
	n := len(cookies)
	ans := 0x7fffffff
	temp := make([]int, n)
	var dfs func(int)
	dfs = func(i int) {
		if i == n {
			maxVal := 0
			for j := range temp {
				if temp[j] > maxVal {
					maxVal = temp[j]
				}
			}
			ans = min(ans, maxVal)
			return
		}
		for j := i + 1; j < n; j++ {
			temp[j] += cookies[j]
			dfs(j + 1)
			temp[j] -= cookies[j]
		}
	}
	dfs(-1)
	return ans
}

func findAndReplacePattern(words []string, pattern string) (rt []string) {
	for _, w := range words {
		m := map[byte]byte{}
		r := map[byte]byte{}
		flag := true
		for i := range w {
			c, ok := m[pattern[i]]
			cc, okk := r[w[i]]
			if ok && okk && c == w[i] && cc == pattern[i] {
				continue
			} else if !ok && !okk {
				m[pattern[i]] = w[i]
				r[w[i]] = pattern[i]
			} else {
				flag = false
				break
			}
		}
		if flag {
			rt = append(rt, w)
		}
	}
	return
}

func findDiagonalOrder(mat [][]int) (rt []int) {
	m, n := len(mat), len(mat[0])
	maxSum := m + n - 2
	d := 1
	valid := func(x, y int) bool {
		return x >= 0 && y >= 0 && x < m && y < n
	}
	for r := 0; r <= maxSum; r++ {
		d = -d
		if d == 1 {
			for i := 0; i <= r; i++ {
				if !valid(i, r-i) {
					break
				}
				rt = append(rt, mat[i][r-i])
			}
		} else {
			for i := r; i >= 0; i-- {
				if !valid(i, r-i) {
					break
				}
				rt = append(rt, mat[i][r-i])
			}
		}
	}
	return
}

func smallestDistancePair(nums []int, k int) int {
	n := len(nums)
	ans := make([]int, 0, n*(n+1)/2)
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			ans = append(ans, abs(nums[i]-nums[j]))
		}
	}
	sort.Ints(ans)
	return ans[k-1]
}

func findPairs(nums []int, k int) int {
	ans := map[[2]int]struct{}{}
	sort.Ints(nums)
	n := len(nums)
	for i := 0; i < n; i++ {
		j := i + sort.Search(n-i, func(j int) bool {
			return nums[i+j] >= nums[i]+k
		})
		if j < n && j != i && nums[j]-nums[i] == k {
			ans[[2]int{nums[i], nums[j]}] = struct{}{}
		}
	}
	return len(ans)
}

func duplicateZeros(arr []int) {
	n := len(arr)
	j := 0
	idx := n - 1
	for i := 0; i < n; i++ {
		if arr[j] == 0 {
			i++
		}
		j++
		if i == n {
			arr[idx] = 0
			idx--
			j--
		}
	}
	for idx >= 0 {
		j--
		arr[idx] = arr[j]
		if arr[j] == 0 {
			arr[idx-1] = 0
			idx--
		}
		idx--
	}
}

func insert(aNode *ListNode, x int) *ListNode {
	t := &ListNode{Val: x}
	if aNode == nil {
		t.Next = t
		return t
	}
	if aNode.Next == aNode {
		t.Next = aNode
		aNode.Next = t
		return aNode
	}
	maxPre := aNode
	p := aNode.Next
	for p != aNode {
		if p.Next.Next.Val < p.Next.Val {
			maxPre = p
		}
		p = p.Next
	}
	if x >= maxPre.Next.Val || x <= maxPre.Next.Next.Val {
		t.Next = maxPre.Next.Next
		maxPre.Next.Next = t
		return aNode
	}
	for i := maxPre.Next.Next; i != maxPre.Next; i = i.Next {
		if x >= i.Val && x <= i.Next.Val {
			t.Next = i.Next
			i.Next = t
			break
		}
	}
	return aNode
}

func greatestLetter(s string) string {
	cntUpper := [26]bool{}
	cntLower := [26]bool{}
	for i := range s {
		if s[i] >= 'a' && s[i] <= 'z' {
			cntLower[s[i]-'a'] = true
		} else {
			cntUpper[s[i]-'A'] = true
		}
	}
	for i := 25; i >= 0; i-- {
		if cntLower[i] && cntUpper[i] {
			return string(byte('A' + i))
		}
	}
	return ""
}

func minimumNumbers(num int, k int) int {
	if num == 0 {
		return 0
	}
	t := num % 10
	for i := 1; i < 10; i++ {
		if t == i*k%10 && i*k <= num {
			return i
		}
	}
	return -1
}

func longestSubsequence(s string, k int) int {
	n := len(s)
	cnt := 0
	var sum int64
	for i := n - 1; i >= 0; i-- {
		if s[i] == '1' {
			if cnt > 31 {
				continue
			}
			if t := sum + (1 << cnt); t <= int64(k) {
				cnt++
				sum = t
			}
		} else {
			cnt++
		}
	}
	return cnt
}

func findFrequentTreeSum(root *TreeNode) (ans []int) {
	type node struct {
		val int
		cnt int
	}
	m := map[int]int{}
	var save []*node
	var dfs func(*TreeNode) int
	dfs = func(cur *TreeNode) int {
		if cur == nil {
			return 0
		}
		sum := dfs(cur.Left) + dfs(cur.Right) + cur.Val
		if idx, ok := m[sum]; ok {
			save[idx].cnt++
		} else {
			t := &node{
				val: sum,
				cnt: 1,
			}
			m[sum] = len(save)
			save = append(save, t)
		}
		return sum
	}
	dfs(root)
	sort.Slice(save, func(i, j int) bool {
		return save[i].cnt > save[j].cnt
	})
	for i := range save {
		if save[i].cnt == save[0].cnt {
			ans = append(ans, save[i].val)
		}
	}
	return
}

func findBottomLeftValue(root *TreeNode) int {
	var queue []*TreeNode
	if root != nil {
		queue = append(queue, root)
	}
	for len(queue) != 0 {
		n := len(queue)
		for i := 0; i < n; i++ {
			cur := queue[i]
			if cur.Left != nil {
				queue = append(queue, cur.Left)
			}
			if cur.Right != nil {
				queue = append(queue, cur.Right)
			}
		}
		if len(queue) == n {
			break
		}
		queue = queue[n:]
	}
	return queue[0].Val
}

func findSubstring(s string, words []string) (ans []int) {
	length := len(words[0])
	n := len(s)
	m := map[string]int{}
	for _, w := range words {
		m[w]++
	}
	total := len(words) * length
r:
	for i := 0; i < n; i++ {
		cnt := map[string]int{}
		for j := i; j < i+total && j+length <= n; j += length {
			if _, ok := m[s[j:j+length]]; ok {
				cnt[s[j:j+length]]++
			} else {
				continue r
			}
		}
		flag := true
		for k, v := range m {
			if c, ok := cnt[k]; !ok || c != v {
				flag = false
				break
			}
		}
		if flag {
			ans = append(ans, i)
		}
	}
	return
}

func largestValues(root *TreeNode) (ans []int) {
	var queue []*TreeNode
	if root == nil {
		return
	}
	queue = append(queue, root)
	for len(queue) > 0 {
		n := len(queue)
		maxVal := -1 << 31
		for i := 0; i < n; i++ {
			cur := queue[i]
			if cur.Left != nil {
				queue = append(queue, cur.Left)
			}
			if cur.Right != nil {
				queue = append(queue, cur.Right)
			}
			if maxVal < cur.Val {
				maxVal = cur.Val
			}
		}
		ans = append(ans, maxVal)
		queue = queue[n:]
	}
	return
}

func minCost(costs [][]int) int {
	n := len(costs)
	for i := 1; i < n; i++ {
		for j := 0; j < 3; j++ {
			costs[i][j] += min(costs[i-1][(j+1)%3], costs[i-1][(j+2)%3])
		}
	}
	return min(min(costs[n-1][0], costs[n-1][1]), costs[n-1][2])
}

func countAsterisks(s string) (ans int) {
	cnt := 0
	for i := range s {
		if s[i] == '|' {
			cnt++
			continue
		}
		if s[i] == '*' && cnt%2 == 0 {
			ans++
		}
	}
	return
}

// TODO
func countPairs(n int, edges [][]int) int64 {
	idx := make([]int, n)
	for i := range idx {
		idx[i] = -1
	}
	getA := func(cur int) (int, int) {
		i := 0
		for idx[cur] >= 0 {
			cur = idx[cur]
			i++
		}
		return cur, i
	}
	for _, e := range edges {
		if idx[e[0]] < 0 && idx[e[1]] < 0 {
			idx[e[1]] += idx[e[0]]
			idx[e[0]] = e[1]
		} else if idx[e[0]] < 0 {
			a, _ := getA(e[1])
			if a != e[0] {
				idx[a] += idx[e[0]]
				idx[e[0]] = a
			}
		} else if idx[e[1]] < 0 {
			a, _ := getA(e[0])
			if a != e[1] {
				idx[a] += idx[e[1]]
				idx[e[1]] = a
			}
		} else {
			a, acnt := getA(e[0])
			b, bcnt := getA(e[1])
			if a != b {
				if acnt > bcnt {
					idx[a] += idx[b]
					idx[b] = a
				} else {
					idx[b] += idx[a]
					idx[a] = b
				}
			}
		}
	}
	cnts := make([]int, 0, n)
	for i := range idx {
		if idx[i] <= -1 {
			cnts = append(cnts, -idx[i])
		}
	}
	if len(cnts) <= 1 {
		return 0
	}
	var ans int64 = 0
	for i := range cnts {
		for j := i + 1; j < len(cnts); j++ {
			ans += int64(cnts[i]) * int64(cnts[j])
		}
	}
	return ans
}

func maximumXOR(nums []int) int {
	cnt := [32]int{}
	for _, v := range nums {
		i := 0
		for v != 0 {
			if (v & 1) == 1 {
				cnt[i]++
			}
			v >>= 1
			i++
		}
	}
	ans := 0
	for i := 31; i >= 0; i-- {
		if cnt[i] > 0 {
			ans += 1 << i
		}
	}
	return ans
}

func checkXMatrix(grid [][]int) bool {
	n := len(grid)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if i == j || i+j == n-1 {
				if grid[i][j] == 0 {
					return false
				}
			} else {
				if grid[i][j] != 0 {
					return false
				}
			}
		}
	}
	return true
}

func countHousePlacements(n int) int {
	dp := make([][2]int, n)
	dp[0][0] = 1
	dp[0][1] = 1
	for i := 1; i < n; i++ {
		dp[i][0] = dp[i-1][1] + dp[i-1][0]
		dp[i][1] = dp[i-1][0]
		dp[i][0] %= mod
		dp[i][1] %= mod
	}
	t := int64((dp[n-1][0] + dp[n-1][1]) % mod)
	return int((t * t) % int64(mod))
}

func maximumsSplicedArray(nums1 []int, nums2 []int) int {
	n := len(nums1)
	div := make([]int, n)
	for i := 0; i < n; i++ {
		div[i] = nums1[i] - nums2[i]
	}
	maxv, minv := math.MinInt32, math.MaxInt32
	sum := 0
	for i := range div {
		sum += div[i]
		if sum < 0 {
			sum = 0
		} else if sum > maxv {
			maxv = sum
		}
	}
	sum = 0
	for i := range div {
		sum += div[i]
		if sum > 0 {
			sum = 0
		} else if sum < minv {
			minv = sum
		}
	}
	sum1, sum2 := 0, 0
	for i := range nums1 {
		sum1 += nums1[i]
		sum2 += nums2[i]
	}
	return max(sum1-minv, sum2+maxv)
}

func findLUSlength(strs []string) int {
	m := map[string]int{}
	var dfs func(int)
	for _, str := range strs {
		tmp := make([]byte, 0, len(str))
		dfs = func(i int) {
			m[string(tmp)]++
			for j := i + 1; j < len(str); j++ {
				tmp = append(tmp, str[j])
				dfs(j)
				tmp = tmp[:len(tmp)-1]
			}
		}
		dfs(-1)
	}
	maxl := -1
	for k, v := range m {
		if v == 1 && len(k) > maxl {
			maxl = len(k)
		}
	}
	return maxl
}

func hammingDistance(x int, y int) int {
	return bits.OnesCount(uint(x ^ y))
}

func mincostTickets(days []int, costs []int) int {
	n := len(days)
	dp := make([]int, days[n-1]+1)
	for i := 0; i < n; i++ {
		dp[days[i]] = -1
	}
	for i := 1; i < len(dp); i++ {
		if dp[i] == 0 {
			dp[i] = dp[i-1]
			continue
		}
		a, b, c := 0x7fffffff, 0x7fffffff, 0x7fffffff
		a = dp[i-1] + costs[0]
		if i-7 >= 0 {
			b = dp[i-7] + costs[1]
		} else {
			b = dp[0] + costs[1]
		}
		if i-30 >= 0 {
			c = dp[i-30] + costs[2]
		} else {
			c = dp[0] + costs[2]
		}
		dp[i] = minOf(a, b, c)
	}
	return dp[len(dp)-1]
}

func diffWaysToCompute(expression string) (rt []int) {
	if n, err := strconv.Atoi(expression); err == nil {
		return append(rt, n)
	}
	for i, c := range expression {
		if c == '+' || c == '-' || c == '*' {
			left := diffWaysToCompute(expression[:i])
			right := diffWaysToCompute(expression[i+1:])
			if c == '+' {
				for j := range left {
					for k := range right {
						rt = append(rt, left[j]+right[k])
					}
				}
			} else if c == '-' {
				for j := range left {
					for k := range right {
						rt = append(rt, left[j]-right[k])
					}
				}
			} else {
				for j := range left {
					for k := range right {
						rt = append(rt, left[j]*right[k])
					}
				}
			}
		}
	}
	return rt
}

func fourSum(nums []int, target int) [][]int {
	ans := map[[4]int]struct{}{}
	sort.Ints(nums)
	n := len(nums)
	for i := 0; i < n; i++ {
		if nums[i] > target>>2 {
			break
		}
		for j := i + 1; j < n; j++ {
			s0 := nums[i] + nums[j]
			if s0 > target>>1 {
				break
			}
			for k := j + 1; k < n; k++ {
				sum := s0 + nums[k]
				if sum+nums[k] > target {
					break
				}
				t := target - sum
				idx := sort.Search(n, func(i0 int) bool {
					return nums[i0] >= t
				})
				for idx <= k {
					idx++
				}
				for idx < n && nums[idx] == t {
					ans[[4]int{nums[i], nums[j], nums[k], t}] = struct{}{}
					idx++
				}
			}
		}
	}
	rt := make([][]int, len(ans))
	i := 0
	for k := range ans {
		rt[i] = k[:]
		i++
	}
	return rt
}

func reverseKGroup(head *ListNode, k int) *ListNode {
	H := &ListNode{}
	p := head
	for i := 0; i < k; i++ {
		if p == nil {
			return head
		}
		p = p.Next
	}
	var reverse func(*ListNode, int) *ListNode
	reverse = func(cur *ListNode, cnt int) *ListNode {
		if cnt == 1 {
			return cur
		}
		rt := reverse(cur.Next, cnt-1)
		cur.Next.Next = cur
		cur.Next = nil
		return rt
	}
	H.Next = reverse(head, k)
	head.Next = reverseKGroup(p, k)
	return H.Next
}

func removeElements(head *ListNode, val int) *ListNode {
	H := &ListNode{Next: head}
	p := H
	for p.Next != nil {
		if p.Next.Val == val {
			p.Next = p.Next.Next
			continue
		}
		p = p.Next
	}
	return H.Next
}

func isIsomorphic(s string, t string) bool {
	s2t := map[byte]byte{}
	t2s := map[byte]byte{}
	for i := range s {
		a, oka := s2t[s[i]]
		b, okb := t2s[t[i]]
		if oka && okb && a == t[i] && b == s[i] {
			continue
		} else if !oka && !okb {
			s2t[s[i]] = t[i]
			t2s[t[i]] = s[i]
		} else {
			return false
		}
	}
	return true
}

func lowestCommonAncestor(root, p, q *TreeNode) (ans *TreeNode) {
	done := false
	var dfs func(cur *TreeNode) bool
	dfs = func(cur *TreeNode) bool {
		if done || cur == nil {
			return false
		}
		l := dfs(cur.Left)
		r := dfs(cur.Right)
		if cur == p || cur == q {
			if l || r {
				done = true
				ans = cur
			}
			return true
		}
		if !done && l && r {
			done = true
			ans = cur
		}
		return l || r
	}
	dfs(root)
	return
}

func binaryTreePaths(root *TreeNode) (rt []string) {
	var buf []string
	var dfs func(*TreeNode)
	dfs = func(cur *TreeNode) {
		if cur.Left == nil && cur.Right == nil {
			if len(buf) != 0 {
				rt = append(rt, strings.Join(buf, "->"))
			}
			return
		}
		if cur.Left != nil {
			buf = append(buf, strconv.Itoa(cur.Left.Val))
			dfs(cur.Left)
			buf = buf[:len(buf)-1]
		}
		if cur.Right != nil {
			buf = append(buf, strconv.Itoa(cur.Right.Val))
			dfs(cur.Right)
			buf = buf[:len(buf)-1]
		}
	}
	if root != nil {
		buf = append(buf, strconv.Itoa(root.Val))
		dfs(root)
	}
	return
}

func isPowerOfFour(n int) bool {
	if cnt := bits.OnesCount(uint(n)); cnt == 1 {
		if z := bits.TrailingZeros(uint(n)); z%2 == 0 {
			return true
		}
	}
	return false
}

func intersect(nums1 []int, nums2 []int) (rt []int) {
	var larger, smaller []int
	if len(nums2) > len(nums1) {
		larger = nums2
		smaller = nums1
	} else {
		larger = nums1
		smaller = nums2
	}
	cnt := map[int]int{}
	for i := range smaller {
		cnt[smaller[i]]++
	}
	for i := range larger {
		if n, ok := cnt[larger[i]]; ok && n > 0 {
			rt = append(rt, larger[i])
			cnt[larger[i]]--
		}
	}
	return
}

func nextGreaterElement(n int) int {
	num := []byte(strconv.Itoa(n))
	flag := false
	for i := len(num) - 1; i > 0; i-- {
		if num[i] > num[i-1] {
			idx := i
			for j := i; j < len(num); j++ {
				if num[idx] > num[j] && num[j] > num[i-1] {
					idx = j
				}
			}
			num[idx], num[i-1] = num[i-1], num[idx]
			sort.Slice(num[i:], func(i0, j0 int) bool {
				return num[i+i0] < num[i+j0]
			})
			flag = true
			break
		}
	}
	if flag {
		rt := 0
		for i := 0; i < len(num); i++ {
			rt *= 10
			rt += int(num[i] - '0')
		}
		if rt > math.MaxInt32 {
			return -1
		}
		return rt
	}
	return -1
}

func firstUniqChar(s string) int {
	idx := [26]int{}
	for i := range idx {
		idx[i] = -1
	}
	for i := range s {
		if idx[s[i]-'a'] == -1 {
			idx[s[i]-'a'] = i
			continue
		}
		idx[s[i]-'a'] = -2
	}
	minv := 1<<31 - 1
	for i := range idx {
		if idx[i] >= 0 && idx[i] < minv {
			minv = idx[i]
		}
	}
	if minv == 1<<31-1 {
		return -1
	}
	return minv
}

func minimumAbsDifference(arr []int) (ans [][]int) {
	sort.Ints(arr)
	n := len(arr)
	minv := math.MaxInt32
	for i := 1; i < n; i++ {
		if arr[i]-arr[i-1] < minv {
			minv = arr[i] - arr[i-1]
		}
	}
	for i := 1; i < n; i++ {
		if arr[i]-arr[i-1] == minv {
			ans = append(ans, []int{arr[i-1], arr[i]})
		}
	}
	return
}

func search(nums []int, target int) int {
	n := len(nums)
	last := nums[n-1]
	k := sort.Search(n, func(i int) bool {
		return nums[i] <= last
	})
	if last == target {
		return n - 1
	}
	if last > target {
		idx := sort.SearchInts(nums[k:], target)
		if idx != n-k && nums[k+idx] == target {
			return k + idx
		}
	} else {
		idx := sort.SearchInts(nums[:k], target)
		if idx != k && nums[idx] == target {
			return idx
		}
	}
	return -1
}

func jump(nums []int) int {
	n := len(nums)
	dp := make([]int, n)
	for i := range dp {
		dp[i] = 0x7fffffff
	}
	dp[0] = 0
	for i := 0; i < n; i++ {
		up := min(i+nums[i], n-1)
		for j := i + 1; j <= up; j++ {
			dp[j] = min(dp[j], dp[i]+1)
		}
	}
	return dp[n-1]
}

func decodeMessage(key string, message string) string {
	trans := [26]byte{}
	idx := byte('a')
	for i := range key {
		if key[i] != ' ' && trans[key[i]-'a'] == 0 {
			trans[key[i]-'a'] = idx
			idx++
		}
	}
	buf := []byte(message)
	for i := range buf {
		if buf[i] != ' ' {
			buf[i] = trans[buf[i]-'a']
		}
	}
	return string(buf)
}

func spiralMatrix(m int, n int, head *ListNode) [][]int {
	matrix := make([][]int, m)
	for i := 0; i < m; i++ {
		matrix[i] = make([]int, n)
		for j := 0; j < n; j++ {
			matrix[i][j] = -1
		}
	}
	var dir = [4]struct{ x, y int }{{0, 1}, {1, 0}, {0, -1}, {-1, 0}}
	p := head
	x, y := 0, 0
	curDir := 0
	cnt := 0
	for p != nil {
		matrix[x][y] = p.Val
		nx, ny := x+dir[curDir].x, y+dir[curDir].y
		if nx < 0 || nx >= m || ny < 0 || ny >= n || matrix[nx][ny] != -1 {
			curDir = (curDir + 1) % 4
			cnt++
			if cnt == 4 {
				break
			}
			continue
		}
		cnt = 0
		x, y = nx, ny
		p = p.Next
	}
	return matrix
}

func peopleAwareOfSecret(n int, delay int, forget int) int {
	type node struct {
		total, inc, able int
	}
	dp := make([]node, n+1)
	dp[1].total = 1
	dp[1].able = 0
	dp[1].inc = 1
	for i := 2; i <= n; i++ {
		dp[i].able = dp[i-1].able
		if i-delay > 0 {
			dp[i].able += dp[i-delay].inc
		}
		if i-forget > 0 {
			dp[i].total -= dp[i-forget].inc
			dp[i].able -= dp[i-forget].inc
		}
		dp[i].total += dp[i-1].total + dp[i].able
		dp[i].inc = dp[i].able
		dp[i].total = ((dp[i].total % mod) + mod) % mod
		dp[i].able = ((dp[i].able % mod) + mod) % mod
		dp[i].inc = ((dp[i].inc % mod) + mod) % mod
	}
	return dp[n].total
}

func replaceWords(dictionary []string, sentence string) string {
	m := make(map[string]struct{}, len(dictionary))
	for i := range dictionary {
		m[dictionary[i]] = struct{}{}
	}
	words := strings.Split(sentence, " ")
	for i := range words {
		for j := 1; j < len(words[i]); j++ {
			if _, ok := m[words[i][:j]]; ok {
				words[i] = words[i][:j]
				break
			}
		}
	}
	return strings.Join(words, " ")
}

func replaceWords1(dictionary []string, sentence string) string {
	type node struct {
		end  bool
		next [26]*node
	}
	root := &node{}
	add := func(s string) {
		p := root
		for i := range s {
			if p.next[s[i]-'a'] == nil {
				p.next[s[i]-'a'] = &node{}
			}
			p = p.next[s[i]-'a']
		}
		p.end = true
	}
	query := func(s string) string {
		p := root
		for i := range s {
			if p.end {
				return s[:i]
			}
			if p.next[s[i]-'a'] == nil {
				break
			}
			p = p.next[s[i]-'a']
		}
		return s
	}
	for i := range dictionary {
		add(dictionary[i])
	}
	words := strings.Split(sentence, " ")
	for i := range words {
		words[i] = query(words[i])
	}
	return strings.Join(words, " ")
}

func minCostToMoveChips(position []int) int {
	oddCnt := 0
	for i := range position {
		if position[i]%2 == 1 {
			oddCnt++
		}
	}
	return min(oddCnt, len(position)-oddCnt)
}

func canReceiveAllSignals(intervals [][]int) bool {
	sort.Slice(intervals, func(i, j int) bool {
		return intervals[i][0] < intervals[j][0]
	})
	n := len(intervals)
	for i := 0; i < n-1; i++ {
		if intervals[i][1] > intervals[i+1][0] {
			return false
		}
	}
	return true
}

func minSwaps(chess []int) int {
	onesCnt := 0
	for i := range chess {
		if chess[i] == 1 {
			onesCnt++
		}
	}
	n := len(chess)
	curCnt := 0
	for i := 0; i < onesCnt; i++ {
		if chess[i] == 1 {
			curCnt++
		}
	}
	ans := curCnt
	for i := onesCnt; i < n; i++ {
		if chess[i] == 1 {
			curCnt++
		}
		if chess[i-onesCnt] == 1 {
			curCnt--
		}
		if curCnt > ans {
			ans = curCnt
		}
	}
	return onesCnt - ans
}

func buildTransferStation(area [][]int) int {
	m := len(area)
	n := len(area[0])
	var points [][2]int
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if area[i][j] == 1 {
				points = append(points, [2]int{i, j})
			}
		}
	}
	ans := 0x7fffffff
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			cur := 0
			for _, p := range points {
				cur += abs(p[0]-i) + abs(p[1]-j)
			}
			if ans > cur {
				ans = cur
			}
		}
	}
	return ans
}

func minTransfers(distributions [][]int) int {
	hosts := [12]int{}
	for _, d := range distributions {
		hosts[d[0]] -= d[2]
		hosts[d[1]] += d[2]
	}
	var opos, oneg []int
	for i := range hosts {
		if hosts[i] > 0 {
			opos = append(opos, hosts[i])
		} else if hosts[i] < 0 {
			oneg = append(oneg, hosts[i])
		}
	}
	ans := 8
	var dfs func([]int, []int, int, int)
	dfs = func(pos []int, neg []int, cnt int, depth int) {
		if depth >= ans {
			return
		}
		if cnt == 0 {
			if ans > depth {
				ans = depth
			}
			return
		}
		for i := range pos {
			if pos[i] == 0 {
				continue
			}
			for j := range neg {
				if neg[j] == 0 {
					continue
				}
				oldp, oldn := pos[i], neg[j]
				sum := oldp + oldn
				if sum > 0 {
					pos[i] = sum
					neg[j] = 0
					dfs(pos, neg, cnt-1, depth+1)
				} else if sum < 0 {
					pos[i] = 0
					neg[j] = sum
					dfs(pos, neg, cnt-1, depth+1)
				} else {
					dfs(pos, neg, cnt-2, depth+1)
				}
				pos[i] = oldp
				neg[j] = oldn
			}
		}
	}
	dfs(opos, oneg, len(opos)+len(oneg), 0)
	return ans
}

func sortedSquares(nums []int) []int {
	mid := sort.SearchInts(nums, 0)
	n := len(nums)
	rt := make([]int, n)
	p, q := mid-1, mid
	for i := 0; i < n; i++ {
		if p >= 0 && q < n {
			if -nums[p] < nums[q] {
				rt[i] = nums[p] * nums[p]
				p--
			} else {
				rt[i] = nums[q] * nums[q]
				q++
			}
		} else if p >= 0 {
			rt[i] = nums[p] * nums[p]
			p--
		} else {
			rt[i] = nums[q] * nums[q]
			q++
		}
	}
	return rt
}

func lenLongestFibSubseq(arr []int) int {
	ans := 0
	n := len(arr)
	var dfs func(int, int, int)
	dfs = func(pre, cur, cnt int) {
		if cnt > 2 && cnt > ans {
			ans = cnt
		}
		next := cur + sort.SearchInts(arr[cur:], arr[pre]+arr[cur])
		if next != n && arr[next] == arr[pre]+arr[cur] {
			dfs(cur, next, cnt+1)
		}
	}
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			dfs(i, j, 2)
		}
	}
	return ans
}

func evaluateTree(root *TreeNode) bool {
	if root.Left == nil && root.Right == nil {
		return root.Val == 1
	}
	l := evaluateTree(root.Left)
	r := evaluateTree(root.Right)
	if root.Val == 2 {
		return l || r
	}
	return l && r
}

//func latestTimeCatchTheBus(buses []int, passengers []int, capacity int) int {
//	buses = append(buses, 0)
//	sort.Ints(buses)
//	sort.Ints(passengers)
//	type bus struct {
//		full        bool
//		firstOneIdx int
//		lastOneIdx  int
//	}
//	status := make([]bus, len(buses))
//	cur := 0
//	for i := 1; i < len(buses); i++ {
//		idx := sort.SearchInts(passengers, buses[i]+1)
//		next := idx
//		if idx-cur >= capacity {
//			next = cur + capacity
//			status[i].full = true
//		}
//		status[i].firstOneIdx = cur
//		status[i].lastOneIdx = next - 1
//		cur = next
//	}
//	for i := len(buses) - 1; i > 0; i-- {
//		if !status[i].full {
//			k := status[i].lastOneIdx
//			for j := buses[i]; j > buses[i-1]; j-- {
//				if j != passengers[k] {
//					return j
//				}
//				if k > 0 {
//					k--
//				}
//			}
//		}
//		if buses[i]-buses[i-1] == capacity {
//			continue
//		}
//		if status[i].lastOneIdx-status[i].firstOneIdx == passengers[status[i].lastOneIdx]-passengers[status[i].firstOneIdx] {
//			continue
//		}
//
//	}
//}
//
//func minSumSquareDiff(nums1 []int, nums2 []int, k1 int, k2 int) int64 {
//	n := len(nums1)
//	div := make([]int, n)
//	for i := range div {
//		div[i] = abs(nums1[i] - nums2[i])
//	}
//	sort.Ints(div)
//	sum := 0
//	for i := range div {
//		sum += div[i]
//	}
//	avg := sum / n
//
//}

func fillCups(amount []int) int {
	cnt := 0
	for {
		sort.Ints(amount)
		if amount[0] == 0 {
			return cnt + max(amount[1], amount[2])
		}
		t := amount[1] - amount[0] + 1
		cnt += t
		amount[1] -= t
		amount[2] -= t
	}
}

func canChange(start string, target string) bool {
	type node struct {
		val byte
		pos int
	}
	n := len(start)
	startPos := make([]node, 0, n)
	targetPos := make([]node, 0, n)
	for i := range start {
		if start[i] != '_' {
			startPos = append(startPos, node{
				val: start[i],
				pos: i,
			})
		}
	}
	for i := range target {
		if target[i] != '_' {
			targetPos = append(targetPos, node{
				val: target[i],
				pos: i,
			})
		}
	}
	if len(startPos) != len(targetPos) {
		return false
	}
	for i := range startPos {
		if startPos[i].val != targetPos[i].val {
			return false
		}
		if startPos[i].val == 'L' && startPos[i].pos < targetPos[i].pos {
			return false
		}
		if startPos[i].val == 'R' && startPos[i].pos > targetPos[i].pos {
			return false
		}
	}
	return true
}

func oddCells(m int, n int, indices [][]int) (ans int) {
	rows, cols := make([]int, m), make([]int, n)
	for _, d := range indices {
		rows[d[0]]++
		cols[d[1]]++
	}
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if (rows[i]+cols[j])%2 == 1 {
				ans++
			}
		}
	}
	return
}

func findRedundantConnection(edges [][]int) []int {
	uf := union_find.InitUnionFind(len(edges) + 1)
	for _, e := range edges {
		if !uf.Union(e[0], e[1]) {
			return e
		}
	}
	return nil
}

func surroundedRegions(board [][]byte) {
	m := len(board)
	n := len(board[0])
	bfs := func(x, y int) {
		var queue [][2]int
		queue = append(queue, [2]int{x, y})
		board[x][y] = 'T'
		for len(queue) > 0 {
			nn := len(queue)
			for i := 0; i < nn; i++ {
				cur := queue[i]
				for _, d := range dir4 {
					nx, ny := cur[0]+d.x, cur[1]+d.y
					if nx >= 0 && nx < m && ny >= 0 && ny < n && board[nx][ny] == 'O' {
						board[nx][ny] = 'T'
						queue = append(queue, [2]int{nx, ny})
					}
				}
			}
			queue = queue[nn:]
		}
	}
	replace := func(old, new byte) {
		for i := range board {
			for j := range board[i] {
				if board[i][j] == old {
					board[i][j] = new
				}
			}
		}
	}
	for i := 0; i < n; i++ {
		if board[0][i] == 'O' {
			bfs(0, i)
		}
		if board[m-1][i] == 'O' {
			bfs(m-1, i)
		}
	}
	for i := 0; i < m; i++ {
		if board[i][0] == 'O' {
			bfs(i, 0)
		}
		if board[i][n-1] == 'O' {
			bfs(i, n-1)
		}
	}
	replace('O', 'X')
	replace('T', 'O')
}

func numIslands(grid [][]byte) int {
	m := len(grid)
	n := len(grid[0])
	bfs := func(x, y int) {
		var queue [][2]int
		queue = append(queue, [2]int{x, y})
		grid[x][y] = 'X'
		for len(queue) > 0 {
			nn := len(queue)
			for i := 0; i < nn; i++ {
				cur := queue[i]
				for _, d := range dir4 {
					nx, ny := cur[0]+d.x, cur[1]+d.y
					if nx >= 0 && nx < m && ny >= 0 && ny < n && grid[nx][ny] == '1' {
						grid[nx][ny] = 'X'
						queue = append(queue, [2]int{nx, ny})
					}
				}
			}
			queue = queue[nn:]
		}
	}
	ans := 0
	for i := range grid {
		for j := range grid[i] {
			if grid[i][j] == '1' {
				ans++
				bfs(i, j)
			}
		}
	}
	return ans
}

func findCircleNum(isConnected [][]int) int {
	n := len(isConnected)
	uf := union_find.InitUnionFind(n)
	for i := range isConnected {
		for j := range isConnected[i] {
			if i != j && isConnected[i][j] == 1 {
				uf.Union(i, j)
			}
		}
	}
	return uf.Groups()
}

func accountsMerge(accounts [][]string) [][]string {
	n := len(accounts)
	uf := union_find.InitUnionFind(n)
	m := map[string]int{}
	for i := range accounts {
		for j := 1; j < len(accounts[i]); j++ {
			if idx, ok := m[accounts[i][j]]; ok {
				uf.Union(idx, i)
			} else {
				m[accounts[i][j]] = i
			}
		}
	}
	temp := map[int][]string{}
	for k, v := range m {
		fa := uf.Find(v)
		if _, ok := temp[fa]; !ok {
			temp[fa] = append(temp[fa], accounts[fa][0])
		}
		temp[fa] = append(temp[fa], k)
	}
	ans := make([][]string, 0, uf.Groups())
	for k := range temp {
		ans = append(ans, temp[k])
	}
	for i := range ans {
		sort.Strings(ans[i][1:])
	}
	return ans
}

func minSwapsCouples(row []int) int {
	n := len(row)
	uf := union_find.InitUnionFind(n / 2)
	for i := 0; i < n; i += 2 {
		uf.Union(row[i]/2, row[i+1]/2)
	}
	return n/2 - uf.Groups()
}

func numberOfPairs(nums []int) []int {
	cnt := [101]int{}
	for i := range nums {
		cnt[nums[i]]++
	}
	a, b := 0, 0
	for i := range cnt {
		if cnt[i]%2 == 1 {
			b += 1
		}
		a += cnt[i] / 2
	}
	return []int{a, b}
}

func maximumSum(nums []int) int {
	m := map[int][]int{}
	f := func(x int) (sum int) {
		for x != 0 {
			sum += x % 10
			x /= 10
		}
		return
	}
	for i := range nums {
		sum := f(nums[i])
		m[sum] = append(m[sum], nums[i])
	}
	ans := -1
	for _, v := range m {
		if len(v) <= 1 {
			continue
		}
		sort.Ints(v)
		if t := v[len(v)-1] + v[len(v)-2]; ans < t {
			ans = t
		}
	}
	return ans
}

func smallestTrimmedNumbers(nums []string, queries [][]int) (ans []int) {
	length := len(nums[0])
	type node struct {
		num *string
		idx int
	}
	arr := make([]*node, len(nums))
	for i := range arr {
		arr[i] = &node{
			num: &nums[i],
			idx: i,
		}
	}
	for i := range queries {
		k, trim := queries[i][0], queries[i][1]
		left := length - trim
		sort.Slice(arr, func(i, j int) bool {
			if (*arr[i].num)[left:] == (*arr[j].num)[left:] {
				return arr[i].idx < arr[j].idx
			}
			return (*arr[i].num)[left:] < (*arr[j].num)[left:]
		})
		ans = append(ans, arr[k-1].idx)
	}
	return
}

func minOperations(nums []int, numsDivide []int) int {
	sort.Ints(nums)
	sort.Ints(numsDivide)
	pre := 0
l:
	for i := range nums {
		if nums[i] != pre {
			pre = nums[i]
			for j := range numsDivide {
				if numsDivide[j]%nums[i] != 0 {
					continue l
				}
			}
			return i
		}
	}
	return -1
}

func arrayNesting(nums []int) int {
	ans := 0
	for i := range nums {
		cnt := 0
		p := i
		for nums[p] >= 0 {
			p, nums[p] = nums[p], -1
			cnt++
		}
		if ans < cnt {
			ans = cnt
		}
	}
	return ans
}

func shiftGrid(grid [][]int, k int) [][]int {
	m, n := len(grid), len(grid[0])
	total := m * n
	f := func(x, y int) (nx, ny int) {
		pos := (x*n + y + k) % total
		return pos / n, pos % n
	}
	ans := make([][]int, m)
	for i := range ans {
		ans[i] = make([]int, n)
	}
	for i := range grid {
		for j := range grid[i] {
			x, y := f(i, j)
			ans[x][y] = grid[i][j]
		}
	}
	return ans
}

func pruneTree(root *TreeNode) *TreeNode {
	if root == nil {
		return nil
	}
	root.Left = pruneTree(root.Left)
	root.Right = pruneTree(root.Right)
	if root.Left == nil && root.Right == nil && root.Val == 0 {
		return nil
	}
	return root
}

func sequenceReconstruction(nums []int, sequences [][]int) bool {
	n := len(nums)
	g := make([][]int, n)
	in := make([]int, n)
	for i := range sequences {
		for j := 1; j < len(sequences[i]); j++ {
			p, q := sequences[i][j-1], sequences[i][j]
			g[p] = append(g[p], q)
			in[q]++
		}
	}
	//for {
	//	for i := range in {
	//		if
	//	}
	//}
	return true
}

func bestHand(ranks []int, suits []byte) string {
	isFlush := func() bool {
		flush := true
		for i := range suits {
			if suits[i] != suits[0] {
				flush = false
				break
			}
		}
		return flush
	}
	isThree := func() bool {
		cnt := map[int]int{}
		for i := range ranks {
			cnt[ranks[i]]++
		}
		if len(cnt) <= 3 {
			for _, v := range cnt {
				if v >= 3 {
					return true
				}
			}
		}
		return false
	}
	isPair := func() bool {
		cnt := map[int]int{}
		for i := range ranks {
			cnt[ranks[i]]++
		}
		if len(cnt) <= 4 {
			for _, v := range cnt {
				if v >= 2 {
					return true
				}
			}
		}
		return false
	}
	isHigh := func() bool {
		cnt := map[int]int{}
		for i := range ranks {
			cnt[ranks[i]]++
		}
		return len(cnt) == 5
	}
	if isFlush() {
		return "Flush"
	}
	if isThree() {
		return "Three of a Kind"
	}
	if isPair() {
		return "Pair"
	}
	if isHigh() {
		return "High Card"
	}
	return ""
}

func zeroFilledSubarray(nums []int) int64 {
	cnt := map[int]int{}
	for i := 0; i < len(nums); i++ {
		if nums[i] == 0 {
			j := 0
			for i+j < len(nums) && nums[i+j] == 0 {
				j++
			}
			if j > 0 {
				cnt[j]++
			}
			i += j
		}
	}
	maxV := 0
	for k := range cnt {
		if k > maxV {
			maxV = k
		}
	}
	if maxV == 0 {
		return 0
	}
	dp := make([]int, maxV+1)
	dp[1] = 1
	for i := 2; i < len(dp); i++ {
		dp[i] = dp[i-1] + i
	}
	ans := 0
	for k, v := range cnt {
		ans += dp[k] * v
	}
	return int64(ans)
}

func shortestSequence(rolls []int, k int) int {
	n := len(rolls)
	cnt := make([]int, k+1)
	vis := make([]bool, k+1)
	for i := range rolls {
		cnt[rolls[i]]++
	}
	ans := math.MaxInt32
	for i := 0; i < n; i++ {
		cnt[rolls[i]]--
		if !vis[rolls[i]] {
			for j := 1; j <= k; j++ {
				ans = min(ans, cnt[j]+1)
			}
			vis[rolls[i]] = true
		}
	}
	for i := 1; i <= k; i++ {
		if !vis[i] {
			return 1
		}
	}
	return ans + 1
}

func repeatedCharacter(s string) byte {
	cnt := [26]int{}
	for i := 0; i < len(s); i++ {
		cnt[s[i]-'a']++
		if cnt[s[i]-'a'] == 2 {
			return s[i]
		}
	}
	return 'a'
}

func equalPairs(grid [][]int) int {
	ans := 0
	n := len(grid)
	for i := range grid {
		for j := 0; j < n; j++ {
			flag := true
			for k := 0; k < n; k++ {
				if grid[i][k] != grid[k][j] {
					flag = false
					break
				}
			}
			if flag {
				ans++
			}
		}
	}
	return ans
}

func countExcellentPairs(nums []int, k int) int64 {
	nums = RemoveDup(nums)
	sort.Slice(nums, func(i, j int) bool {
		return bits.OnesCount(uint(nums[i])) < bits.OnesCount(uint(nums[j]))
	})
	ans := 0
	for i := range nums {
		t := sort.Search(len(nums), func(j int) bool {
			return bits.OnesCount(uint(nums[i]&nums[j]))+bits.OnesCount(uint(nums[i]|nums[j])) >= k
		})
		ans += len(nums) - t
	}
	return int64(ans)
}

func distanceBetweenBusStops(distance []int, start int, destination int) int {
	for i := 1; i < len(distance); i++ {
		distance[i] += distance[i-1]
	}
	total := distance[len(distance)-1]
	if start > destination {
		start, destination = destination, start
	}
	t := distance[destination] - distance[start]
	return min(total-t, t)
}

func fractionAddition(expression string) string {
	ans := Frac{0, 1}
	a := strings.Split(expression, "+")
	for i := range a {
		t := Frac{0, 1}
		b := strings.Split(a[i], "-")
		if b[0] != "" {
			f := strings.Split(b[0], "/")
			u, _ := strconv.Atoi(f[0])
			d, _ := strconv.Atoi(f[1])
			t.Add(Frac{u, d})
		}
		for j := 1; j < len(b); j++ {
			f := strings.Split(b[j], "/")
			u, _ := strconv.Atoi(f[0])
			d, _ := strconv.Atoi(f[1])
			t.Sub(Frac{u, d})
		}
		ans.Add(t)
	}
	return fmt.Sprintf("%d/%d", ans.up, ans.down)
}

func arrayRankTransform(arr []int) []int {
	n := len(arr)
	idx := Nums(0, n)
	sort.Slice(idx, func(i, j int) bool {
		return arr[idx[i]] < arr[idx[j]]
	})
	rt := make([]int, n)
	r := 1
	for i := range idx {
		if i > 0 && arr[idx[i]] != arr[idx[i-1]] {
			r++
		}
		rt[idx[i]] = r
	}
	return rt
}

func validSquare(p1 []int, p2 []int, p3 []int, p4 []int) bool {
	points := [4][]int{p1, p2, p3, p4}
	cnt := map[int]int{}
	for i := 0; i < 4; i++ {
		for j := i + 1; j < 4; j++ {
			cnt[getDistSq(points[i][0]-points[j][0], points[i][1]-points[j][1])]++
		}
	}
	if len(cnt) == 2 {
		kv := MapItems(cnt)
		if kv[0][0] > kv[1][0] {
			kv[0], kv[1] = kv[1], kv[0]
		}
		if kv[0][1] == 4 && kv[1][1] == 2 {
			if kv[0][0]*2 == kv[1][0] {
				return true
			}
		}
	}
	return false
}

func largestComponentSize(nums []int) int {
	const maxNums = 1e5 + 10
	n := len(nums)
	uf := union_find.InitUnionFind(maxNums)
	for i := 0; i < n; i++ {
		for j := 2; j*j <= nums[i]; j++ {
			if nums[i]%j == 0 {
				uf.Union(nums[i], j)
				uf.Union(nums[i], nums[i]/j)
			}
		}
	}
	cnt := map[int]int{}
	for _, v := range nums {
		cnt[uf.Find(v)]++
	}
	ans := 0
	for _, v := range cnt {
		ans = max(ans, v)
	}
	return ans
}

func maxLevelSum(root *TreeNode) int {
	ans := 1
	maxV := math.MinInt32
	cnt := 1
	var queue []*TreeNode
	queue = append(queue, root)
	for len(queue) > 0 {
		n := len(queue)
		sum := 0
		for i := 0; i < n; i++ {
			sum += queue[i].Val
			if queue[i].Left != nil {
				queue = append(queue, queue[i].Left)
			}
			if queue[i].Right != nil {
				queue = append(queue, queue[i].Right)
			}
		}
		if sum > maxV {
			ans = cnt
			maxV = sum
		}
		cnt++
		queue = queue[n:]
	}
	return ans
}

func minimumOperations(nums []int) int {
	getMin := func(arr []int) int {
		rt := 1000
		for i := range arr {
			if arr[i] > 0 && arr[i] < rt {
				rt = arr[i]
			}
		}
		return rt
	}
	cnt := 0
	for {
		minV := getMin(nums)
		flag := false
		for i := range nums {
			if nums[i] > 0 {
				nums[i] -= minV
				flag = true
			}
		}
		if !flag {
			break
		}
		cnt++
	}
	return cnt
}

func maximumGroups(grades []int) int {
	n := len(grades)
	ans := 1
	for ans*(ans+1) <= n*2 {
		ans++
	}
	return ans - 1
}

func closestMeetingNode(edges []int, node1 int, node2 int) int {
	n := len(edges)
	f := func(node int) ([]int, map[int]int) {
		idx := make([]int, 0, n)
		dist := make(map[int]int, n)
		vis := make([]bool, n)
		cnt := 0
		for node != -1 && !vis[node] {
			vis[node] = true
			idx = append(idx, node)
			dist[node] = cnt
			cnt++
			node = edges[node]
		}
		return idx, dist
	}
	idx1, dist1 := f(node1)
	idx2, dist2 := f(node2)
	common := Intersection(idx1, idx2)
	ans := math.MaxInt32
	minV := math.MaxInt32
	for i := range common {
		cur := max(dist1[common[i]], dist2[common[i]])
		if cur < minV {
			ans = common[i]
			minV = cur
		} else if cur == minV && ans > common[i] {
			ans = common[i]
		}
	}
	return Cond(ans == math.MaxInt32, -1, ans).(int)
}

func longestCycle(edges []int) int {
	n := len(edges)
	ans := 0
	vis := make([]bool, n)
	for i := range edges {
		dist := map[int]int{}
		node := i
		cnt := 0
		for node != -1 {
			if vis[node] {
				d, ok := dist[node]
				if ok {
					ans = max(ans, cnt-d)
				}
				break
			}
			vis[node] = true
			dist[node] = cnt
			cnt++
			node = edges[node]
		}
	}
	return Cond(ans == 0, -1, ans).(int)
}

func generateTheString(n int) string {
	rt := bytes.Repeat([]byte{'a'}, n)
	if n%2 == 0 {
		rt[0] = 'b'
	}
	return Bytes2Str(rt)
}

func videoStitching(clips [][]int, time int) int {
	ts := make([]int, time+1)
	for _, c := range clips {
		if c[0] < time {
			ts[c[0]] = max(ts[c[0]], min(c[1], time))
		}
	}
	cnt := 0
	begin, end := 0, 0
	for end < time {
		newEnd := end
		for j := begin; j <= end; j++ {
			newEnd = max(newEnd, ts[j])
		}
		if newEnd == end {
			return -1
		}
		begin, end = end, newEnd
		cnt++
	}
	return Cond(end == time, cnt, -1).(int)
}

func orderlyQueue(s string, k int) string {
	n := len(s)
	ans := Str2Bytes(s)
	if k == 1 {
		b := make([]byte, n*2)
		copy(b, s)
		copy(b[n:], s)
		for i := 0; i < n; i++ {
			if bytes.Compare(ans, b[i:i+n]) > 0 {
				ans = b[i : i+n]
			}
		}
	} else {
		sort.Slice(ans, func(i, j int) bool {
			return ans[i] < ans[j]
		})
	}
	return Bytes2Str(ans)
}

func minSubsequence(nums []int) (rt []int) {
	h := heaps.IntHeap(nums)
	heap.Init(&h)
	sum := 0
	total := SummarizingInt(nums)
	for {
		top := heap.Pop(&h).(int)
		sum += top
		total -= top
		rt = append(rt, top)
		if sum > total {
			break
		}
	}
	return
}

func minSubsequence1(nums []int) []int {
	sum := 0
	total := SummarizingInt(nums)
	sort.Sort(sort.Reverse(sort.IntSlice(nums)))
	for i := range nums {
		sum += nums[i]
		total -= nums[i]
		if sum > total {
			return nums[:i+1]
		}
	}
	return nums
}

func dayOfYear(date string) int {
	d, _ := time.Parse("2006-01-02", date)
	return d.YearDay()
}

func countNumbersWithUniqueDigits(n int) int {
	if n == 0 {
		return 1
	}
	return C(1, 9)*A(n-1, 9) + countNumbersWithUniqueDigits(n-1)
}

func minTimeToType(word string) int {
	ans := 0
	cur := 'a'
	for _, c := range word {
		if cur == c {
			ans++
			continue
		}
		a := abs(int(c - cur))
		b := 26 - a
		ans += min(a, b) + 1
		cur = c
	}
	return ans
}

func checkString(s string) bool {
	return !strings.Contains(s, "ba")
}

func stringMatching(words []string) (rt []string) {
	n := len(words)
	vis := make([]bool, n)
	for i := range words {
		if vis[i] {
			continue
		}
		for j := range words {
			if vis[j] {
				continue
			}
			if i != j && strings.Contains(words[i], words[j]) {
				vis[j] = true
				rt = append(rt, words[j])
			}
		}
	}
	return
}

func mergeSimilarItems(items1 [][]int, items2 [][]int) [][]int {
	m := make(map[int]int, len(items1)+len(items2))
	for _, it := range items1 {
		m[it[0]] += it[1]
	}
	for _, it := range items2 {
		m[it[0]] += it[1]
	}
	rt := make([][]int, 0, len(m))
	for k, v := range m {
		rt = append(rt, []int{k, v})
	}
	sort.Slice(rt, func(i, j int) bool {
		return rt[i][0] < rt[j][0]
	})
	return rt
}

func countBadPairs(nums []int) int64 {
	if len(nums) < 2 {
		return 0
	}
	cnt := map[int]int{}
	ans := 0
	for i := range nums {
		cnt[nums[i]-i]++
	}
	for _, v := range cnt {
		if v >= 2 {
			ans += C(2, v)
		}
	}
	return int64(C(2, len(nums)) - ans)
}

func taskSchedulerII(tasks []int, space int) int64 {
	last := map[int]int{}
	cur := 1
	for i := range tasks {
		if n, ok := last[tasks[i]]; ok {
			if cur-n <= space {
				cur = n + space + 1
			}
		}
		last[tasks[i]] = cur
		cur++
	}
	return int64(cur - 1)
}

func minimumReplacement(nums []int) int64 {
	ans := 0
	minV := nums[len(nums)-1]
	for i := len(nums) - 1; i >= 0; i-- {
		minV = min(minV, nums[i])
		if nums[i] > minV {
			ans += Ceil(nums[i], minV) - 1
			if nums[i]%minV != 0 {
				for j := 2; j < minV; j++ {
					if nums[i]/i < minV {
						minV = nums[i] / i
						break
					}
				}
			}
		}
	}
	return int64(ans)
}
