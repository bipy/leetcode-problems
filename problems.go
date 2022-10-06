package leetcode

import (
	"bytes"
	"fmt"
	"github.com/emirpasic/gods/trees/redblacktree"
	"leetcode/template/heaps"
	"leetcode/template/partial_sum"
	"leetcode/template/segment_tree"
	"leetcode/template/sparse_table"
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

// 819 最常见的单词
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

// 剑指 Offer 03 数组中重复的数字
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

// 剑指 Offer 06 从尾到头打印链表
func reversePrint(head *ListNode) []int {
	if head == nil {
		return []int{}
	}
	rt := reversePrint(head.Next)
	rt = append(rt, head.Val)
	return rt
}

// 剑指 Offer 11 旋转数组的最小数字
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

// 剑指 Offer 17 打印从1到最大的n位数
func printNumbers(n int) []int {
	rt := make([]int, int(math.Pow10(n)-1))
	for i := 0; i < len(rt); i++ {
		rt[i] = i + 1
	}
	return rt
}

// 剑指 Offer 21 调整数组顺序使奇数位于偶数前面
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

// 剑指 Offer 24 反转链表
func reverseList(head *ListNode) *ListNode {
	if head == nil || head.Next == nil {
		return head
	}
	rt := reverseList(head.Next)
	head.Next.Next = head
	head.Next = nil
	return rt
}

// 剑指 Offer 27 二叉树的镜像
func mirrorTree(root *TreeNode) *TreeNode {
	if root == nil {
		return nil
	}
	mirrorTree(root.Left)
	mirrorTree(root.Right)
	root.Left, root.Right = root.Right, root.Left
	return root
}

// 386 字典序排数
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

// 剑指 Offer 29 顺时针打印矩阵
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

// 剑指 Offer 32 - II 从上到下打印二叉树 II
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

// 824 山羊拉丁文
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

// 面试题 05.06 整数转换
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

// 105 从前序与中序遍历序列构造二叉树
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

// 958 二叉树的完全性检验
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

// 62 不同路径
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

// 1289 下降路径最小和 II
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

// 576 出界的路径数
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

// 2255 统计是给定字符串前缀的字符串数目
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

// 2256 最小平均差
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

// 2257 统计网格图中没有被保卫的格子数
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

// 16 最接近的三数之和
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

// 1305 两棵二叉搜索树中的所有元素
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

// 2259 移除指定数字得到的最大结果
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

// 2260 必须拿起的最小连续卡牌数
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

// 2261 含最多 K 个可整除元素的子数组
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

// 2262 字符串的总引力 TODO
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

// 937 重新排列日志文件
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

// 2248 多个数组求交集
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

// 2249 统计圆内格点数目
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

// 2250 统计包含每个点的矩形数目
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

// 433 最小基因变化
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

// 2264 字符串中最大的 3 位相同数字
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

// 2265 统计值等于子树平均值的节点数
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

// 2266 统计打字方案数
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

// 2267 检查是否有合法括号字符串路径 TODO
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

// 2243 计算字符串的数字和
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

// 2244 完成所有任务需要的最少轮数
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

// 2245 转角路径的乘积中最多能有几个尾随零 TODO
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

// 2269 找到一个数字的 K 美丽值
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

// 2270 分割数组的方案数
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

// 2271 毯子覆盖的最多白色砖块数
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

// 2272 最大波动的子字符串 TODO
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

// 812 最大三角形面积
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

// 953 验证外星语词典
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

// 2273 移除字母异位词后的结果数组
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

// 2274 不含特殊楼层的最大连续楼层数
func maxConsecutive(bottom int, top int, special []int) int {
	sort.Ints(special)
	ans := special[0] - bottom
	for i := 1; i < len(special); i++ {
		ans = max(ans, special[i]-special[i-1]-1)
	}
	return max(ans, top-special[len(special)-1])
}

// 2275 按位与结果大于零的最长组合
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

// 462 最少移动次数使数组元素相等 II
func minMoves2(nums []int) int {
	sort.Ints(nums)
	target := nums[len(nums)>>1]
	ans := 0
	for _, v := range nums {
		ans += abs(v - target)
	}
	return ans
}

// 436 寻找右区间
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

// 961 在长度 2N 的数组中找出重复 N 次的元素
func repeatedNTimes(nums []int) int {
	n := len(nums)
	for {
		x, y := rand.Intn(n), rand.Intn(n)
		if x != y && nums[x] == nums[y] {
			return nums[x]
		}
	}
}

// 2278 字母在字符串中的百分比
func percentageLetter(s string, letter byte) int {
	cnt := 0
	for i := range s {
		if s[i] == letter {
			cnt++
		}
	}
	return cnt * 100 / len(s)
}

// 2279 装满石头的背包的最大数量
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

// 2280 表示一个折线图的最少线段数
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

// 965 单值二叉树
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

// 467 环绕字符串中唯一的子字符串
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

// 1021 删除最外层的括号
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

// 2283 判断一个数的数字计数是否等于数位的值
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

// 2284 最多单词数的发件人
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

// 2285 道路的最大总重要性
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

// 468 验证IP地址
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

// 2287 重排字符形成目标字符串
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

// 2288 价格减免
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

// 2290 到达角落需要移除障碍物的最小数目
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

// 1022 从根到叶的二进制数之和
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

// 剑指 Offer II 114 外星文字典
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

// 450 删除二叉搜索树中的节点
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

// 2177 找到和为给定整数的三个连续整数
func sumOfThree(num int64) []int64 {
	if num%3 == 0 {
		return []int64{num/3 - 1, num / 3, num/3 + 1}
	}
	return []int64{}
}

// 829 连续整数求和
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

// 929 独特的电子邮件地址
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

// 1037 有效的回旋镖
func isBoomerang(points [][]int) bool {
	x1, y1 := points[0][0], points[0][1]
	x2, y2 := points[1][0], points[1][1]
	x3, y3 := points[2][0], points[2][1]
	return x1*(y2-y3)-x2*(y1-y3)+x3*(y1-y2) != 0
}

// 875 爱吃香蕉的珂珂
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

// 926 将字符串翻转到单调递增
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

// 2303 计算应缴税款总额
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

// 2304 网格中的最小路径代价
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

// 2305 公平分发饼干
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

// 890 查找和替换模式
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

// 498 对角线遍历
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

// 719 找出第 K 小的数对距离
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

// 532 数组中的 k-diff 数对
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

// 1089 复写零
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

// 剑指 Offer II 029 排序的循环链表
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

// 2309 兼具大小写的最好英文字母
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

// 2310 个位数字为 K 的整数之和
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

// 2311 小于等于 K 的最长二进制子序列
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

// 508 出现次数最多的子树元素和
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

// 513 找树左下角的值
// 剑指 Offer II 045 二叉树最底层最左边的值
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

// 30 串联所有单词的子串
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

// 515 在每个树行中找最大值
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

// 剑指 Offer II 091 粉刷房子
func minCost(costs [][]int) int {
	n := len(costs)
	for i := 1; i < n; i++ {
		for j := 0; j < 3; j++ {
			costs[i][j] += min(costs[i-1][(j+1)%3], costs[i-1][(j+2)%3])
		}
	}
	return min(min(costs[n-1][0], costs[n-1][1]), costs[n-1][2])
}

// 2315 统计星号
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

// 2317 操作后的最大异或和
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

// 2319 判断矩阵是否是一个 X 矩阵
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

// 2320 统计放置房子的方式数
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

// 2321 拼接数组的最大分数
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

// 128 最长连续序列
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

// 461 汉明距离
func hammingDistance(x int, y int) int {
	return bits.OnesCount(uint(x ^ y))
}

// 983 最低票价
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

// 241 为运算表达式设计优先级
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

// 18 四数之和
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

// 25 K 个一组翻转链表
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

// 203 移除链表元素
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

// 205 同构字符串
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

// 235 二叉搜索树的最近公共祖先
// 236 二叉树的最近公共祖先
// 剑指 Offer 68 - II 二叉树的最近公共祖先
func lowestCommonAncestor(root, p, q *TreeNode) (ans *TreeNode) {
	done := false
	var dfs func(cur *TreeNode) bool
	dfs = func(cur *TreeNode) bool {
		if done || cur == nil {
			return false
		}
		l := dfs(cur.Left)
		r := dfs(cur.Right)
		if done {
			return false
		}
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

// 257 二叉树的所有路径
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

// 342 4的幂
func isPowerOfFour(n int) bool {
	if cnt := bits.OnesCount(uint(n)); cnt == 1 {
		if z := bits.TrailingZeros(uint(n)); z%2 == 0 {
			return true
		}
	}
	return false
}

// 350 两个数组的交集 II
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

// 556 下一个更大元素 III
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

// 387 字符串中的第一个唯一字符
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

// 1200 最小绝对差
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

// 33 搜索旋转排序数组
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

// 45 跳跃游戏 II
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

// 2325 解密消息
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

// 2326 螺旋矩阵 IV
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

// 2327 知道秘密的人数
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

// 648 单词替换
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

// 648 单词替换
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

// 1217 玩筹码
func minCostToMoveChips(position []int) int {
	oddCnt := 0
	for i := range position {
		if position[i]%2 == 1 {
			oddCnt++
		}
	}
	return min(oddCnt, len(position)-oddCnt)
}

// 未来城竞赛
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

// 未来城竞赛
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

// 未来城竞赛
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

// 未来城竞赛
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

// 977 有序数组的平方
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

// 873 最长的斐波那契子序列的长度
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

// 2331 计算布尔二叉树的值
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

// TODO
func latestTimeCatchTheBus(buses []int, passengers []int, capacity int) int {
	buses = append(buses, 0)
	sort.Ints(buses)
	sort.Ints(passengers)
	type bus struct {
		full        bool
		firstOneIdx int
		lastOneIdx  int
	}
	status := make([]bus, len(buses))
	cur := 0
	for i := 1; i < len(buses); i++ {
		idx := sort.SearchInts(passengers, buses[i]+1)
		next := idx
		if idx-cur >= capacity {
			next = cur + capacity
			status[i].full = true
		}
		status[i].firstOneIdx = cur
		status[i].lastOneIdx = next - 1
		cur = next
	}
	for i := len(buses) - 1; i > 0; i-- {
		if !status[i].full {
			k := status[i].lastOneIdx
			for j := buses[i]; j > buses[i-1]; j-- {
				if j != passengers[k] {
					return j
				}
				if k > 0 {
					k--
				}
			}
		}
		if buses[i]-buses[i-1] == capacity {
			continue
		}
		if status[i].lastOneIdx-status[i].firstOneIdx == passengers[status[i].lastOneIdx]-passengers[status[i].firstOneIdx] {
			continue
		}

	}
	return 0
}

// TODO
func minSumSquareDiff(nums1 []int, nums2 []int, k1 int, k2 int) int64 {
	n := len(nums1)
	div := make([]int, n)
	for i := range div {
		div[i] = abs(nums1[i] - nums2[i])
	}
	sort.Ints(div)
	sum := 0
	for i := range div {
		sum += div[i]
	}
	_ = sum / n
	return 0
}

// 2335 装满杯子需要的最短总时长
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

// 2337 移动片段得到字符串
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

// 1252 奇数值单元格的数目
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

// 684 冗余连接
func findRedundantConnection(edges [][]int) []int {
	uf := union_find.InitUnionFind(len(edges) + 1)
	for _, e := range edges {
		if !uf.Union(e[0], e[1]) {
			return e
		}
	}
	return nil
}

// 130 被围绕的区域
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

// 200 岛屿数量
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

// 547 省份数量
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

// 721 账户合并
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

// 765 情侣牵手
func minSwapsCouples(row []int) int {
	n := len(row)
	uf := union_find.InitUnionFind(n / 2)
	for i := 0; i < n; i += 2 {
		uf.Union(row[i]/2, row[i+1]/2)
	}
	return n/2 - uf.Groups()
}

// 2341 数组能形成多少数对
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

// 2342 数位和相等数对的最大和
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

// 2343 裁剪数字后查询第 K 小的数字
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

// 2344 使数组可以被整除的最少删除次数
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

// 565 数组嵌套
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

// 1260 二维网格迁移
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

// 814 二叉树剪枝
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

// 剑指 Offer II 115 重建序列
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

// 2347 最好的扑克手牌
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

// 2348 全 0 子数组的数目
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

// 2350 不可能得到的最短骰子序列 TODO
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

// 2351 第一个出现两次的字母
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

// 2352 相等行列对
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

// 2354 优质数对的数目
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

// 1184 公交站间的距离
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

// 592 分数加减运算
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

// 1331 数组序号转换
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

// 593 有效的正方形
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

// 952 按公因数计算最大组件大小
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

// 1161 最大层内元素和
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

// 2357 使数组中所有元素都等于零
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

// 2358 分组的最大数量
func maximumGroups(grades []int) int {
	n := len(grades)
	ans := 1
	for ans*(ans+1) <= n*2 {
		ans++
	}
	return ans - 1
}

// 2359 找到离给定两个节点最近的节点
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

// 2360 图中的最长环
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

// 1374 生成每种字符都是奇数个的字符串
func generateTheString(n int) string {
	rt := bytes.Repeat([]byte{'a'}, n)
	if n%2 == 0 {
		rt[0] = 'b'
	}
	return Bytes2Str(rt)
}

// 1024 视频拼接
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

// 899 有序队列
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

// 1403 非递增顺序的最小子序列
func minSubsequence(nums []int) (rt []int) {
	h := heaps.InitIntMinHeap(nums)
	sum := 0
	total := SummarizingInt(nums)
	for {
		top := h.Pop()
		sum += top
		total -= top
		rt = append(rt, top)
		if sum > total {
			break
		}
	}
	return
}

// 1403 非递增顺序的最小子序列
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

// 1154 一年中的第几天
func dayOfYear(date string) int {
	d, _ := time.Parse("2006-01-02", date)
	return d.YearDay()
}

// 357 统计各位数字都不同的数字个数
func countNumbersWithUniqueDigits(n int) int {
	if n == 0 {
		return 1
	}
	return C(1, 9)*A(n-1, 9) + countNumbersWithUniqueDigits(n-1)
}

// 1974 使用特殊打字机键入单词的最少时间
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

// 2124 检查是否所有 A 都在 B 之前
func checkString(s string) bool {
	return !strings.Contains(s, "ba")
}

// 1408 数组中的字符串匹配
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

// 2363 合并相似的物品
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

// 2364 统计坏数对的数目
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

// 2365 任务调度器 II
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

// 2366 TODO
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

// 2367 算术三元组的数目
func arithmeticTriplets(nums []int, diff int) int {
	ans := 0
	for i := range nums {
		for j := i + 1; j < len(nums); j++ {
			if nums[j]-nums[i] == diff {
				for k := j + 1; k < len(nums); k++ {
					if nums[k]-nums[j] == diff {
						ans++
					}
				}
			}
		}
	}
	return ans
}

// 2368 受限条件下可到达节点的数目
func reachableNodes(n int, edges [][]int, restricted []int) int {
	uf := union_find.InitUnionFind(n)
	m := map[int]struct{}{}
	for i := range restricted {
		m[restricted[i]] = struct{}{}
	}
	for _, e := range edges {
		if _, ok := m[e[0]]; ok {
			continue
		}
		if _, ok := m[e[1]]; ok {
			continue
		}
		uf.Union(e[0], e[1])
	}
	return uf.GroupSize(0)
}

// 2369 检查数组是否存在有效划分
func validPartition(nums []int) bool {
	n := len(nums)
	if n == 2 {
		return nums[0] == nums[1]
	}
	dp := make([]bool, n+1)
	dp[0] = true
	for i := 2; i <= n; i++ {
		if nums[i-1] == nums[i-2] && dp[i-2] {
			dp[i] = true
		}
		if i >= 3 && nums[i-1] == nums[i-2] && nums[i-1] == nums[i-3] && dp[i-3] {
			dp[i] = true
		}
		if i >= 3 && nums[i-1] == nums[i-2]+1 && nums[i-1] == nums[i-3]+2 && dp[i-3] {
			dp[i] = true
		}
	}
	return dp[n]
}

// 2370 最长理想子序列
func longestIdealString(s string, k int) int {
	dp := [26]int{}
	for _, c := range s {
		begin := max(0, int(c-'a')-k)
		end := min(int(c-'a')+k, 25)
		dp[c-'a'] = maxOf(dp[begin:end+1]...) + 1
	}
	return maxOf(dp[:]...)
}

// 636 函数的独占时间
func exclusiveTime(n int, logs []string) []int {
	var stack []int
	cost := make([]int, n)
	last := 0
	for _, log := range logs {
		sp := strings.Split(log, ":")
		id, _ := strconv.Atoi(sp[0])
		t, _ := strconv.Atoi(sp[2])
		if sp[1] == "end" {
			t++
		}
		if len(stack) > 0 {
			cost[stack[len(stack)-1]] += t - last
			last = t
		}
		if sp[1] == "start" {
			stack = append(stack, id)
		} else {
			stack = stack[:len(stack)-1]
		}
	}
	return cost
}

// 761 特殊的二进制序列
func makeLargestSpecial(s string) string {
	cnt, i := 0, 0
	var queue []string
	for j := 0; j < len(s); j++ {
		if s[j] == '1' {
			cnt++
		} else {
			cnt--
		}
		if cnt == 0 {
			queue = append(queue, "1"+makeLargestSpecial(s[i+1:j])+"0")
			i = j + 1
		}
	}
	sort.Sort(sort.Reverse(sort.StringSlice(queue)))
	return strings.Join(queue, "")
}

// 107 二叉树的层序遍历 II
func levelOrderBottom(root *TreeNode) (rt [][]int) {
	if root == nil {
		return [][]int{}
	}
	queue := []*TreeNode{root}
	for len(queue) > 0 {
		n := len(queue)
		level := make([]int, n)
		for i := 0; i < n; i++ {
			level[i] = queue[i].Val
			if queue[i].Left != nil {
				queue = append(queue, queue[i].Left)
			}
			if queue[i].Right != nil {
				queue = append(queue, queue[i].Right)
			}
		}
		rt = append(rt, level)
		queue = queue[n:]
	}
	for i := len(rt)/2 - 1; i >= 0; i-- {
		j := len(rt) - i - 1
		rt[i], rt[j] = rt[j], rt[i]
	}
	return
}

// 2302 统计得分小于 K 的子数组数目
func countSubarrays(nums []int, k int64) int64 {
	ans, left, sum := 0, 0, 0
	for right, num := range nums {
		sum += num
		for sum*(right-left+1) >= int(k) {
			sum -= nums[left]
			left++
		}
		ans += right - left + 1
	}
	return int64(ans)
}

// 1232 缀点成线
func checkStraightLine(coordinates [][]int) bool {
	return SameLine(coordinates)
}

// 1413 逐步求和得到正数的最小值
func minStartValue(nums []int) int {
	for i := 1; i < len(nums); i++ {
		nums[i] += nums[i-1]
	}
	return max(1, 1-minOf(nums...))
}

// 833 字符串中的查找与替换
func findReplaceString(s string, indices []int, sources []string, targets []string) string {
	arr := strings.Split(s, "")
	for i := range sources {
		idx := indices[i]
		if len(s[idx:]) >= len(sources[i]) {
			if s[idx:idx+len(sources[i])] == sources[i] {
				arr[idx] = targets[i]
				for j := idx + 1; j < idx+len(sources[i]); j++ {
					arr[j] = ""
				}
			}
		}
	}
	return strings.Join(arr, "")
}

// 1578 使绳子变成彩色的最短时间
func minCost1(colors string, neededTime []int) (ans int) {
	pre := 0
	for i := 1; i <= len(colors); i++ {
		if i == len(colors) || colors[i] != colors[pre] {
			if i-pre > 1 {
				ans += SummarizingInt(neededTime[pre:i]) - maxOf(neededTime[pre:i]...)
			}
			pre = i
		}
	}
	return
}

// 1536 排布二进制网格的最少交换次数
func minSwaps1(grid [][]int) (ans int) {
	n := len(grid)
	type node struct {
		idx, val int
	}
	trailingZeros := make([]node, n)
	for i := range grid {
		for j := n - 1; j >= -1; j-- {
			if j == -1 || grid[i][j] != 0 {
				trailingZeros[i] = node{
					idx: i,
					val: n - j - 1,
				}
				break
			}
		}
	}
	for i := 0; i < n; i++ {
		target := n - i - 1
		ok := false
		for j := 0; j < n; j++ {
			if trailingZeros[j].val >= target {
				trailingZeros[j].val = -1
				ans += trailingZeros[j].idx - i
				ok = true
				break
			}
			trailingZeros[j].idx++
		}
		if !ok {
			return -1
		}
	}
	return
}

// 1537 最大得分
func maxSum(nums1 []int, nums2 []int) int {
	a, b := 0, 0
	p, q := 0, 0
	for p < len(nums1) && q < len(nums2) {
		if nums1[p] < nums2[q] {
			a += nums1[p]
			p++
		} else if nums1[p] > nums2[q] {
			b += nums2[q]
			q++
		} else {
			a = max(a, b) + nums1[p]
			b = a
			p++
			q++
		}
		a %= mod
		b %= mod
	}
	for p < len(nums1) {
		a += nums1[p]
		a %= mod
		p++
	}
	for q < len(nums2) {
		b += nums2[q]
		b %= mod
		q++
	}
	return max(a, b)
}

// 640 求解方程
func solveEquation(equation string) string {
	a, b := 0, 0
	sig := 1
	check := func(it []byte, c int) {
		var n int
		if it[len(it)-1] == 'x' {
			if len(it) == 1 {
				n = 1
			} else {
				n, _ = strconv.Atoi(Bytes2Str(it[:len(it)-1]))
			}
			a += sig * c * n
		} else {
			n, _ = strconv.Atoi(Bytes2Str(it))
			b += sig * c * n
		}
	}
	process := func(part []byte) {
		for _, p := range bytes.Split(part, []byte{'+'}) {
			items := bytes.Split(p, []byte{'-'})
			if len(items[0]) != 0 {
				check(items[0], 1)
			}
			for i := 1; i < len(items); i++ {
				check(items[i], -1)
			}
		}
	}
	bs := Str2Bytes(equation)
	eq := bytes.Split(bs, []byte{'='})
	process(eq[0])
	sig = -1
	process(eq[1])
	if a == 0 {
		if b == 0 {
			return "Infinite solutions"
		}
		return "No solution"
	}
	return fmt.Sprintf("x=%d", -b/a)
}

// 2044 统计按位或能得到最大值的子集数目
func countMaxOrSubsets(nums []int) (ans int) {
	target := 0
	for _, num := range nums {
		target |= num
	}
	var dfs func(i, cur int)
	dfs = func(i, cur int) {
		cur |= nums[i]
		if cur == target {
			ans++
		}
		for j := i + 1; j < len(nums); j++ {
			dfs(j, cur)
		}
	}
	dfs(0, 0)
	return
}

// 785 判断二分图
func isBipartite(graph [][]int) bool {
	uf := union_find.InitUnionFind(len(graph))
	for i := range graph {
		for j := range graph[i] {
			if uf.Find(i) == uf.Find(graph[i][j]) {
				return false
			}
			uf.Union(graph[i][0], graph[i][j])
		}
	}
	return true
}

// 670 最大交换
func maximumSwap(num int) int {
	s := []byte(strconv.Itoa(num))
	n := len(s)
	maxNum := make([]byte, n)
	copy(maxNum, s)
	for i := n - 2; i >= 0; i-- {
		if maxNum[i] < maxNum[i+1] {
			maxNum[i] = maxNum[i+1]
		}
	}
l:
	for i := 0; i < n; i++ {
		if s[i] != maxNum[i] {
			for j := i + 1; j <= n; j++ {
				if j == n || maxNum[j] < maxNum[i] {
					s[j-1], s[i] = s[i], s[j-1]
					break l
				}
			}
		}
	}
	rt, _ := strconv.Atoi(string(s))
	return rt
}

// 1753 移除石子的最大得分
func maximumScore(a int, b int, c int) int {
	nums := [3]int{a, b, c}
	sort.Ints(nums[:])
	if nums[0]+nums[1] >= nums[2] {
		return nums[2] + (nums[0]+nums[1]-nums[2])/2
	}
	return nums[0] + nums[1]
}

// 1849 将字符串拆分为递减的连续值
func splitString(s string) bool {
	ans := false
	var dfs func(i int, target int)
	dfs = func(i int, target int) {
		if ans || i == len(s) {
			ans = true
			return
		}
		for j := i + 1; j <= len(s); j++ {
			if n, _ := strconv.Atoi(s[i:j]); n == target {
				dfs(j, target-1)
			} else if n > target {
				break
			}
		}
	}
	for i := 1; i < len(s); i++ {
		if n, _ := strconv.Atoi(s[:i]); n > 0 {
			dfs(i, n-1)
		}
	}
	return ans
}

// 1616 分割两个字符串得到回文串
func checkPalindromeFormation(a string, b string) bool {
	equal := func(s, rev string) bool {
		for i := 0; i < len(s); i++ {
			if s[i] != rev[len(s)-i-1] {
				return false
			}
		}
		return true
	}
	check := func(x, y string) bool {
		for i := len(x)/2 - 1; i >= 0; i-- {
			if x[i] != x[len(x)-i-1] {
				return equal(x[:i+1], y[len(y)-i-1:]) || equal(y[:i+1], x[len(y)-i-1:])
			}
		}
		return true
	}
	return check(a, b) || check(b, a)
}

// 1541 平衡括号字符串的最少插入次数
func minInsertions(s string) (ans int) {
	left := 0
	for i := 0; i < len(s); i++ {
		if s[i] == '(' {
			left++
		} else if s[i] == ')' {
			if i+1 >= len(s) || s[i+1] != ')' {
				ans++
			} else {
				i++
			}
			if left == 0 {
				ans++
			} else {
				left--
			}
		}
	}
	return ans + left*2
}

// 1417 重新格式化字符串
func reformat(s string) string {
	var a, b []byte
	for i := range s {
		if s[i] >= 'a' && s[i] <= 'z' {
			a = append(a, s[i])
		} else {
			b = append(b, s[i])
		}
	}
	if len(b) > len(a) {
		a, b = b, a
	}
	if len(a)-len(b) > 1 {
		return ""
	}
	sb := strings.Builder{}
	for i := 0; i < len(b); i++ {
		sb.WriteByte(a[i])
		sb.WriteByte(b[i])
	}
	if len(a) > len(b) {
		sb.WriteByte(a[len(a)-1])
	}
	return sb.String()
}

// 1282 用户分组
func groupThePeople(groupSizes []int) (ans [][]int) {
	m := map[int][]int{}
	for i := range groupSizes {
		m[groupSizes[i]] = append(m[groupSizes[i]], i)
	}
	for k, v := range m {
		for i := 0; i < len(v); i += k {
			ans = append(ans, v[i:i+k])
		}
	}
	return
}

// 768 最多能完成排序的块 II
func maxChunksToSorted(arr []int) (ans int) {
	n := len(arr)
	lmax, rmin := make([]int, n), make([]int, n)
	lmax[0], rmin[n-1] = arr[0], arr[n-1]
	for i := 1; i < n; i++ {
		lmax[i] = max(lmax[i-1], arr[i])
		rmin[n-i-1] = min(rmin[n-i], arr[n-i-1])
	}
	for i := 0; i < n; i++ {
		if arr[i] >= lmax[i] && arr[i] <= rmin[i] {
			ans++
		}
	}
	return
}

// 2373 矩阵中的局部最大值
func largestLocal(grid [][]int) [][]int {
	n := len(grid)
	rt := make([][]int, n-2)
	for i := range rt {
		rt[i] = make([]int, n-2)
		for j := range rt[i] {
			for k := 0; k < 3; k++ {
				for l := 0; l < 3; l++ {
					rt[i][j] = max(rt[i][j], grid[i+k][j+l])
				}
			}
		}
	}
	return rt
}

// 2374 边积分最高的节点
func edgeScore(edges []int) int {
	cnt := make([]int, len(edges))
	for i, e := range edges {
		cnt[e] += i
	}
	return maxIdx(cnt)
}

// 2375 根据模式串构造最小数字
func smallestNumber(pattern string) string {
	n := len(pattern)
	ans := ""
	used := [10]bool{}
	var dfs func(cur []byte, i int)
	dfs = func(cur []byte, i int) {
		t := Bytes2Str(cur)
		if ans != "" && ans < t {
			return
		}
		if i == n {
			if ans == "" || ans > t {
				ans = string(cur)
			}
			return
		}
		for j := byte('1'); j <= '9'; j++ {
			if !used[j-'0'] {
				if pattern[i] == 'I' && j > cur[len(cur)-1] || pattern[i] == 'D' && j < cur[len(cur)-1] {
					used[j-'0'] = true
					cur = append(cur, j)
					dfs(cur, i+1)
					cur = cur[:len(cur)-1]
					used[j-'0'] = false
				}
			}
		}
	}
	for j := byte('1'); j <= '9'; j++ {
		used[j-'0'] = true
		dfs([]byte{j}, 0)
		used[j-'0'] = false
	}
	return ans
}

// 2376 TODO
func countSpecialNumbers(n int) int {
	return 0
}

// 1422 分割字符串的最大得分
func maxScore(s string) (ans int) {
	l, r := 0, 0
	for _, c := range s {
		if c == '1' {
			r++
		}
	}
	for i := 0; i < len(s)-1; i++ {
		if s[i] == '1' {
			r--
		} else {
			l++
		}
		ans = max(ans, l+r)
	}
	return ans
}

// 793 阶乘函数后 K 个零
func preimageSizeFZF(k int) int {
	f := func(i int) bool {
		sum := 0
		for j := 5; j <= i; j *= 5 {
			sum += i / j
		}
		return sum >= k
	}
	left := sort.Search(1<<31, f)
	k++
	right := sort.Search(1<<31, f)
	return right - left
}

// 1302 层数最深叶子节点的和
func deepestLeavesSum(root *TreeNode) (ans int) {
	queue := []*TreeNode{root}
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
		ans = sum
		queue = queue[n:]
	}
	return
}

// 1224 最大相等频率
func maxEqualFreq(nums []int) (ans int) {
	cnt := map[int]int{}
	rev := map[int]map[int]struct{}{}
	for i := range nums {
		n, ok := cnt[nums[i]]
		if ok {
			delete(rev[n], nums[i])
			if len(rev[n]) == 0 {
				delete(rev, n)
			}
		}
		cnt[nums[i]] = n + 1
		if _, ook := rev[n+1]; !ook {
			rev[n+1] = map[int]struct{}{}
		}
		rev[n+1][nums[i]] = struct{}{}
		if len(rev) == 2 {
			items := [2][2]int{}
			j := 0
			for k, v := range rev {
				items[j] = [2]int{k, len(v)}
				j++
			}
			if items[0][0]-items[1][0] == 1 && items[0][1] == 1 ||
				items[1][0]-items[0][0] == 1 && items[1][1] == 1 ||
				(items[0][0] == 1 && items[0][1] == 1 || items[1][0] == 1 && items[1][1] == 1) {
				ans = i + 1
			}
		} else if len(rev) == 1 {
			for k, v := range rev {
				if k == 1 || len(v) == 1 {
					ans = i + 1
				}
			}
		}
	}
	return
}

// 1450 在既定时间做作业的学生人数
func busyStudent(startTime []int, endTime []int, queryTime int) (ans int) {
	for i := range startTime {
		if queryTime >= startTime[i] && queryTime <= endTime[i] {
			ans++
		}
	}
	return
}

// 56 合并区间
func merge(intervals [][]int) (ans [][]int) {
	sort.Slice(intervals, func(i, j int) bool {
		if intervals[i][0] == intervals[j][0] {
			return intervals[i][1] > intervals[j][1]
		}
		return intervals[i][0] < intervals[j][0]
	})
	intervals = append(intervals, []int{1 << 31})
	left, right := intervals[0][0], intervals[0][1]
	for i := 1; i < len(intervals); i++ {
		if intervals[i][0] < right {
			if intervals[i][1] > right {
				right = intervals[i][1]
			}
		} else {
			ans = append(ans, []int{left, right})
			left, right = intervals[i][0], intervals[i][1]
		}
	}
	return
}

// 654 最大二叉树
func constructMaximumBinaryTree(nums []int) *TreeNode {
	if len(nums) == 0 {
		return nil
	}
	idx := maxIdx(nums)
	return &TreeNode{
		Val:   nums[idx],
		Left:  constructMaximumBinaryTree(nums[:idx]),
		Right: constructMaximumBinaryTree(nums[idx+1:]),
	}
}

// 2379 得到 K 个黑块的最少涂色次数
func minimumRecolors(blocks string, k int) int {
	cnt := 0
	for i := 0; i < k; i++ {
		if blocks[i] == 'W' {
			cnt++
		}
	}
	ans := cnt
	for i := k; i < len(blocks); i++ {
		if blocks[i] == 'B' && blocks[i-k] == 'W' {
			cnt--
		} else if blocks[i] == 'W' && blocks[i-k] == 'B' {
			cnt++
		}
		if cnt < ans {
			ans = cnt
		}
	}
	return ans
}

// 2380 二进制字符串重新安排顺序需要的时间
func secondsToRemoveOccurrences(s string) int {
	cnt := 0
	for strings.Contains(s, "01") {
		cnt++
		s = strings.ReplaceAll(s, "01", "10")
	}
	return cnt
}

// 2381 字母移位 II
func shiftingLetters(s string, shifts [][]int) string {
	div := make([]int, len(s)+1)
	for _, sh := range shifts {
		if sh[2] == 0 {
			div[sh[0]]--
			div[sh[1]+1]++
		} else if sh[2] == 1 {
			div[sh[0]]++
			div[sh[1]+1]--
		}
	}
	b := []byte(s)
	sum := 0
	for i := range b {
		sum += div[i]
		b[i] = 'a' + byte((int(b[i]-'a')+2600000+sum)%26)
	}
	return string(b)
}

// 2382 删除操作后的最大子段和
func maximumSegmentSum(nums []int, removeQueries []int) []int64 {
	type tuple struct {
		left, right, sum int
	}
	sum := make([]int, len(nums))
	sum[0] = nums[0]
	for i := 1; i < len(sum); i++ {
		sum[i] = sum[i-1] + nums[i]
	}
	getSum := func(l, r int) int {
		if l == 0 {
			return sum[r]
		}
		return sum[r] - sum[l-1]
	}
	rbt := redblacktree.NewWithIntComparator()
	h := heaps.InitHeap(nil, func(i, j interface{}) bool {
		return i.(*tuple).sum > j.(*tuple).sum
	})
	ans := make([]int64, len(nums))
	rbt.Put(0, &tuple{0, len(nums) - 1, sum[len(sum)-1]})
	h.Push(rbt.Left().Value)
	for i, v := range removeQueries {
		node, _ := rbt.Floor(v)
		left, right := node.Key.(int), node.Value.(*tuple).right
		rbt.Remove(left)
		if left < v {
			t := &tuple{left, v - 1, getSum(left, v-1)}
			rbt.Put(left, t)
			h.Push(t)
		}
		if v < right {
			t := &tuple{v + 1, right, getSum(v+1, right)}
			rbt.Put(v+1, t)
			h.Push(t)
		}
		for {
			if h.Size() == 0 {
				ans[i] = 0
				break
			}
			cur := h.Top().(*tuple)
			res, ok := rbt.Get(cur.left)
			if ok && res.(*tuple).right == cur.right {
				ans[i] = int64(cur.sum)
				break
			}
			h.Pop()
		}
	}
	return ans
}

// 2383 赢得比赛需要的最少训练时长
func minNumberOfHours(initialEnergy int, initialExperience int, energy []int, experience []int) int {
	ans := 0
	for i := range experience {
		if initialEnergy <= energy[i] {
			ans += energy[i] - initialEnergy + 1
			initialEnergy = 1
		} else {
			initialEnergy -= energy[i]
		}
		if initialExperience <= experience[i] {
			ans += experience[i] - initialExperience + 1
			initialExperience = experience[i] + 1 + experience[i]
		} else {
			initialExperience += experience[i]
		}
	}
	return ans
}

// 2384 最大回文数字
func largestPalindromic(num string) string {
	cnt := [10]int{}
	for i := range num {
		cnt[num[i]-'0']++
	}
	var seq []byte
	sb := strings.Builder{}
	for i := 9; i >= 0; i-- {
		seq = append(seq, byte('0'+i))
		sb.Write(bytes.Repeat([]byte{byte('0' + i)}, cnt[i]/2))
	}
	for i := 9; i >= 0; i-- {
		if cnt[i]%2 == 1 {
			sb.WriteByte(byte('0' + i))
			break
		}
	}
	for i := len(seq) - 1; i >= 0; i-- {
		sb.Write(bytes.Repeat([]byte{seq[i]}, cnt[seq[i]-'0']/2))
	}
	s := sb.String()
	s = strings.TrimLeft(s, "0")
	s = strings.TrimRight(s, "0")
	if s == "" {
		return "0"
	}
	return s
}

// 2385 感染二叉树需要的总时间
func amountOfTime(root *TreeNode, start int) int {
	g := map[int][]int{}
	queue := []*TreeNode{root}
	for len(queue) > 0 {
		n := len(queue)
		for i := 0; i < n; i++ {
			if queue[i].Left != nil {
				a, b := queue[i].Val, queue[i].Left.Val
				g[a] = append(g[a], b)
				g[b] = append(g[b], a)
				queue = append(queue, queue[i].Left)
			}
			if queue[i].Right != nil {
				a, b := queue[i].Val, queue[i].Right.Val
				g[a] = append(g[a], b)
				g[b] = append(g[b], a)
				queue = append(queue, queue[i].Right)
			}
		}
		queue = queue[n:]
	}
	ans := -1
	q := []int{start}
	vis := map[int]struct{}{}
	vis[start] = struct{}{}
	for len(q) > 0 {
		ans++
		n := len(q)
		for i := 0; i < n; i++ {
			for _, to := range g[q[i]] {
				if _, ok := vis[to]; !ok {
					vis[to] = struct{}{}
					q = append(q, to)
				}
			}
		}
		q = q[n:]
	}
	return ans
}

// 2386 找出数组的第 K 大和
func kSum(nums []int, k int) int64 {
	tot := 0
	for i := range nums {
		if nums[i] >= 0 {
			tot += nums[i]
		} else {
			nums[i] = -nums[i]
		}
	}
	if k == 1 {
		return int64(tot)
	}
	sort.Ints(nums)
	type node struct {
		sum, idx int
	}
	h := heaps.InitHeap(nil, func(i, j interface{}) bool {
		return i.(node).sum < j.(node).sum
	})
	h.Push(node{nums[0], 0})
	for i := 2; i < k; i++ {
		cur := h.Pop().(node)
		if cur.idx+1 < len(nums) {
			h.Push(node{cur.sum + nums[cur.idx+1], cur.idx + 1})
			h.Push(node{cur.sum - nums[cur.idx] + nums[cur.idx+1], cur.idx + 1})
		}
	}
	kth := h.Top().(node)
	return int64(tot - kth.sum)
}

// 1455 检查单词是否为句中其他单词的前缀
func isPrefixOfWord(sentence string, searchWord string) int {
	for i, w := range strings.Split(sentence, " ") {
		if strings.HasPrefix(w, searchWord) {
			return i
		}
	}
	return -1
}

// 655 输出二叉树
func printTree(root *TreeNode) [][]string {
	height := 0
	queue := []*TreeNode{root}
	for len(queue) > 0 {
		n := len(queue)
		for i := 0; i < n; i++ {
			if queue[i].Left != nil {
				queue = append(queue, queue[i].Left)
			}
			if queue[i].Right != nil {
				queue = append(queue, queue[i].Right)
			}
		}
		queue = queue[n:]
		height++
	}
	length := 1<<height - 1
	res := make([][]string, height)
	for i := range res {
		res[i] = make([]string, length)
	}
	var dfs func(cur *TreeNode, x, y int)
	dfs = func(cur *TreeNode, x, y int) {
		res[x][y] = strconv.Itoa(cur.Val)
		if cur.Left != nil {
			dfs(cur.Left, x+1, y-(1<<(height-x-2)))
		}
		if cur.Right != nil {
			dfs(cur.Right, x+1, y+(1<<(height-x-2)))
		}
	}
	dfs(root, 0, (length-1)/2)
	return res
}

// 852 山脉数组的峰顶索引
func peakIndexInMountainArray(arr []int) int {
	return sort.Search(len(arr)-1, func(i int) bool {
		return arr[i] > arr[i+1]
	})
}

// 1460 通过翻转子数组使两个数组相等
func canBeEqual(target []int, arr []int) bool {
	cnt := [1001]int{}
	for i := range target {
		cnt[target[i]]++
		cnt[arr[i]]--
	}
	for i := range cnt {
		if cnt[i] != 0 {
			return false
		}
	}
	return true
}

// 658 找到 K 个最接近的元素
// TODO
func findClosestElements(arr []int, k int, x int) []int {
	n := len(arr)
	idx := sort.Search(n, func(i int) bool {
		return arr[i] >= x
	})
	i, j := idx-1, idx
	for j-i < k && i >= 0 && j < n {
		d1, d2 := x-arr[i], arr[j]-x
		if d1 > d2 {
			j++
		} else {
			i--
		}
	}
	if i < 0 {
		return arr[:k]
	} else if j >= n {
		return arr[n-k:]
	}
	return arr[i+1 : j]
}

// 662 二叉树最大宽度
func widthOfBinaryTree(root *TreeNode) (ans int) {
	levels := map[int]int{}
	var dfs func(cur *TreeNode, flag int, cnt int)
	dfs = func(cur *TreeNode, flag int, cnt int) {
		if cur == nil {
			return
		}
		n, ok := levels[cnt]
		if !ok {
			levels[cnt] = flag
			n = flag
		}
		flag = flag - n + 1
		ans = max(ans, flag)
		dfs(cur.Left, flag<<1, cnt+1)
		dfs(cur.Right, flag<<1|1, cnt+1)
	}
	dfs(root, 0, 0)
	return
}

// 1464 数组中两元素的最大乘积
func maxProduct(nums []int) int {
	first, second := 0, 0
	for i := range nums {
		if nums[i] >= first {
			first, second = nums[i], first
		} else if nums[i] > second {
			second = nums[i]
		}
	}
	return (first - 1) * (second - 1)
}

// 1470 重新排列数组
func shuffle(nums []int, n int) []int {
	ans := make([]int, n<<1)
	for i := 0; i < n; i++ {
		ans[i<<1] = nums[i]
		ans[i<<1|1] = nums[i+n]
	}
	return ans
}

// 998 最大二叉树 II
func insertIntoMaxTree(root *TreeNode, val int) *TreeNode {
	if root == nil || root.Val < val {
		return &TreeNode{
			Val:  val,
			Left: root,
		}
	}
	root.Right = insertIntoMaxTree(root.Right, val)
	return root
}

// 946 验证栈序列
func validateStackSequences(pushed []int, popped []int) bool {
	var stack []int
	j, n := 0, len(pushed)
	for i := 0; i < n; i++ {
		stack = append(stack, pushed[i])
		for len(stack) > 0 && stack[len(stack)-1] == popped[j] {
			stack = stack[:len(stack)-1]
			j++
		}
	}
	return len(stack) == 0
}

// 1475 商品折扣后的最终价格
func finalPrices(prices []int) []int {
	n := len(prices)
	for i := 0; i < n; i++ {
		for j := i + 1; j < len(prices); j++ {
			if prices[j] <= prices[i] {
				prices[i] -= prices[j]
				break
			}
		}
	}
	return prices
}

// 687 最长同值路径
func longestUnivaluePath(root *TreeNode) int {
	ans := 1
	var dfs func(cur *TreeNode) int
	dfs = func(cur *TreeNode) int {
		if cur == nil {
			return 0
		}
		l := dfs(cur.Left)
		r := dfs(cur.Right)
		if cur.Left != nil && cur.Val == cur.Left.Val && cur.Right != nil && cur.Val == cur.Right.Val {
			ans = max(ans, l+r+1)
			return max(l, r) + 1
		}
		if cur.Left != nil && cur.Val == cur.Left.Val {
			ans = max(ans, l+1)
			return l + 1
		}
		if cur.Right != nil && cur.Val == cur.Right.Val {
			ans = max(ans, r+1)
			return r + 1
		}
		return 1
	}
	if root != nil {
		dfs(root)
	}
	return ans - 1
}

// 646 最长数对链
func findLongestChain(pairs [][]int) int {
	n := len(pairs)
	sort.Slice(pairs, func(i, j int) bool {
		if pairs[i][1] == pairs[j][1] {
			return pairs[i][0] < pairs[j][0]
		}
		return pairs[i][1] < pairs[j][1]
	})
	dp := make([]int, n)
	pre := 0
	for i := range pairs {
		idx := sort.Search(i, func(i0 int) bool {
			return pairs[i0][1] >= pairs[i][0]
		})
		if idx > pre {
			dp[i] = maxOf(dp[:idx]...) + 1
		}
	}
	return maxOf(dp[pre:]...) + 1
}

// 2395 和相等的子数组
func findSubarrays(nums []int) bool {
	m := map[int]struct{}{}
	for i := 1; i < len(nums); i++ {
		sum := nums[i] + nums[i-1]
		if _, ok := m[sum]; ok {
			return true
		} else {
			m[sum] = struct{}{}
		}
	}
	return false
}

// 2396 严格回文的数字
func isStrictlyPalindromic(n int) bool {
	f := func(t, b int) (rt []int) {
		for {
			rt = append(rt, t%b)
			t = t / b
			if t == 0 {
				break
			}
		}
		return
	}
	for b := 2; b <= n-2; b++ {
		s := f(n, b)
		for i := 0; i < len(s)/2; i++ {
			if s[i] != s[len(s)-i-1] {
				return false
			}
		}
	}
	return true
}

// 2397 被列覆盖的最多行数
func maximumRows(mat [][]int, cols int) (ans int) {
	m, n := len(mat), len(mat[0])
	nums := make([]int, m)
	for i := range nums {
		t := 0
		for j := 0; j < n; j++ {
			t <<= 1
			t += mat[i][j]
		}
		nums[i] = t
	}
	var dfs func(flag, zeros, length int)
	dfs = func(flag, zeros, length int) {
		if length > n || zeros > cols {
			return
		}
		if length >= n && zeros == cols {
			cnt := 0
			for i := range nums {
				if nums[i]&flag == 0 {
					cnt++
				}
			}
			ans = max(ans, cnt)
			return
		}
		dfs(flag<<1, zeros+1, length+1)
		dfs(flag<<1|1, zeros, length+1)
	}
	dfs(1, 0, 0)
	return
}

// 2398 预算内的最多机器人数目
func maximumRobots(chargeTimes []int, runningCosts []int, budget int64) int {
	n := len(chargeTimes)
	st := sparse_table.InitSparseTable(chargeTimes, func(a, b int) int {
		return max(a, b)
	})
	ps := partial_sum.InitPartialSum(runningCosts)
	idx := sort.Search(n, func(k int) bool {
		minV := int(budget) + 1
		for i := 0; i+k < n; i++ {
			minV = min(minV, st.Query(i, i+k+1)+(k+1)*ps.Query(i, i+k))
		}
		return minV > int(budget)
	})
	return idx
}

// 1582 二进制矩阵中的特殊位置
func numSpecial(mat [][]int) (ans int) {
	rowCnt, colCnt := make([]int, len(mat)), make([]int, len(mat[0]))
	for i := range mat {
		for j := range mat[i] {
			if mat[i][j] == 1 {
				rowCnt[i]++
				colCnt[j]++
			}
		}
	}
	for i := range mat {
		for j := range mat[i] {
			if mat[i][j] == 1 && rowCnt[i] == 1 && colCnt[j] == 1 {
				ans++
			}
		}
	}
	return
}

// 652 寻找重复的子树
func findDuplicateSubtrees(root *TreeNode) (ans []*TreeNode) {
	type node struct {
		ptr *TreeNode
		idx int
	}
	m := map[[3]int]*node{}
	i := 0
	var dfs func(cur *TreeNode) int
	dfs = func(cur *TreeNode) int {
		if cur == nil {
			return 0
		}
		tup := [3]int{cur.Val, dfs(cur.Left), dfs(cur.Right)}
		if n, ok := m[tup]; ok {
			if n.ptr != nil {
				ans = append(ans, n.ptr)
				n.ptr = nil
			}
			return n.idx
		}
		i++
		m[tup] = &node{cur, i}
		return i
	}
	dfs(root)
	return
}

// 828 统计子串中的唯一字符
func uniqueLetterString(s string) (ans int) {
	idx := [26][]int{}
	for i := range idx {
		idx[i] = append(idx[i], -1)
	}
	for i, c := range s {
		idx[c-'A'] = append(idx[c-'A'], i)
	}
	for i := range idx {
		idx[i] = append(idx[i], len(s))
	}
	for i := range idx {
		for j := 1; j < len(idx[i])-1; j++ {
			ans += (idx[i][j] - idx[i][j-1]) * (idx[i][j+1] - idx[i][j])
		}
	}
	return
}

// 1592 重新排列单词间的空格
func reorderSpaces(text string) string {
	cnt := 0
	words := strings.FieldsFunc(text, func(r rune) bool {
		if r == ' ' {
			cnt++
			return true
		}
		return false
	})
	if len(words) == 1 {
		return words[0] + strings.Repeat(" ", cnt)
	}
	m, n := cnt%(len(words)-1), cnt/(len(words)-1)
	if m == 0 {
		return strings.Join(words, strings.Repeat(" ", n))
	}
	return strings.Join(words, strings.Repeat(" ", n)) + strings.Repeat(" ", m)
}

// 667 优美的排列 II
func constructArray(n int, k int) []int {
	arr := Nums(1, n+1)
	for i := 1; i <= k; i++ {
		if i%2 == 0 {
			arr[i] = i/2 + 1
		} else {
			arr[i] = k + 1 - arr[i-1]
		}
	}
	return arr
}

// 1598 文件夹操作日志搜集器
func minOperations1(logs []string) int {
	var stack []string
	for _, log := range logs {
		if log == "../" {
			if len(stack) != 0 {
				stack = stack[:len(stack)-1]
			}
		} else if log != "./" {
			stack = append(stack, log)
		}
	}
	return len(stack)
}

// 669 修剪二叉搜索树
func trimBST(root *TreeNode, low int, high int) *TreeNode {
	if root == nil {
		return nil
	}
	if root.Val < low {
		return trimBST(root.Right, low, high)
	}
	if root.Val > high {
		return trimBST(root.Left, low, high)
	}
	root.Left = trimBST(root.Left, low, high)
	root.Right = trimBST(root.Right, low, high)
	return root
}

// 1608 特殊数组的特征值
func specialArray(nums []int) int {
	sort.Ints(nums)
	n := len(nums)
	x := 1 + sort.Search(n, func(i int) bool {
		return n-sort.SearchInts(nums, i+1) <= i+1
	})
	if x == n-sort.SearchInts(nums, x) {
		return x
	}
	return -1
}

// 2404 出现最频繁的偶数元素
func mostFrequentEven(nums []int) int {
	nums = FilterInt(nums, func(i int) bool {
		return i%2 == 0
	})
	m := CountInt(nums)
	ans, cnt := -1, -1
	for k, v := range m {
		if v > cnt || v == cnt && k < ans {
			cnt = v
			ans = k
		}
	}
	return ans
}

// 2405 子字符串的最优划分
func partitionString(s string) int {
	cnt := [26]bool{}
	ans := 0
	for _, c := range s {
		if cnt[c-'a'] {
			ans++
			cnt = [26]bool{}
		}
		cnt[c-'a'] = true
	}
	return ans + 1
}

// 2406 将区间分为最少组数
func minGroups(intervals [][]int) int {
	div := make([]int, 1e6+10)
	for i := range intervals {
		div[intervals[i][0]]++
		div[intervals[i][1]+1]--
	}
	ans := 0
	cnt := 0
	for i := range div {
		cnt += div[i]
		ans = max(ans, cnt)
	}
	return ans
}

// 2407 最长递增子序列 II
func lengthOfLIS(nums []int, k int) int {
	st := segment_tree.InitSegmentTree(make([]int, maxOf(nums...)+1), max)
	for _, v := range nums {
		maxV := st.Query(max(0, v-k), v-1)
		st.Update(v, maxV+1)
	}
	return st.QueryAll()
}

// 1711 大餐计数
func countPairs1(deliciousness []int) (ans int) {
	m := CountInt(deliciousness)
	items := MapItems(m)
	sort.Slice(items, func(i, j int) bool {
		return items[i][0] < items[j][0]
	})
	maxV := maxOf(deliciousness...) << 1
	for _, it := range items {
		k, v := it[0], it[1]
		for i := 0; maxV>>i != 0; i++ {
			if k == 1<<i && v > 1 {
				ans += C(2, v)
				ans %= mod
			}
			if k < 1<<i && k<<1 != 1<<i {
				if cnt, ok := m[1<<i-k]; ok && cnt > 0 {
					ans += v * cnt
					ans %= mod
				}
			}
		}
		m[k] = 0
	}
	return
}

// 322 零钱兑换
func coinChange(coins []int, amount int) int {
	dp := IntRepeat(math.MaxInt, amount+1)
	dp[0] = 0
	for i := 1; i <= amount; i++ {
		for _, coin := range coins {
			if i >= coin {
				dp[i] = min(dp[i], dp[i-coin]+1)
			}
		}
	}
	if dp[amount] == math.MaxInt {
		return -1
	}
	return dp[amount]
}

// 1619 删除某些元素后的数组均值
func trimMean(arr []int) float64 {
	sort.Ints(arr)
	return AverageInt(arr[len(arr)/20 : len(arr)-len(arr)/20])
}

// TODO:850 矩形面积 II
func rectangleArea(rectangles [][]int) int {
	return 0
}

// 1624 两个相同字符之间的最长子字符串
func maxLengthBetweenEqualCharacters(s string) int {
	ans := -1
	first := IntRepeat(-1, 26)
	for i := range s {
		if idx := first[s[i]-'a']; idx != -1 {
			ans = max(ans, i-idx-1)
		} else {
			first[s[i]-'a'] = i
		}
	}
	return ans
}

func countDaysTogether(arriveAlice string, leaveAlice string, arriveBob string, leaveBob string) int {
	trans := func(date string) int {
		m, _ := strconv.Atoi(date[:2])
		d, _ := strconv.Atoi(date[3:])
		return time.Date(2022, time.Month(m), d, 0, 0, 0, 0, time.UTC).YearDay()
	}
	aa, la := trans(arriveAlice), trans(leaveAlice)
	ab, lb := trans(arriveBob), trans(leaveBob)
	from, to := max(aa, ab), min(la, lb)
	if from > to {
		return 0
	}
	return to - from + 1
}

func matchPlayersAndTrainers(players []int, trainers []int) int {
	sort.Ints(players)
	sort.Ints(trainers)
	ans := 0
	for i := range players {
		idx := sort.Search(len(trainers), func(i0 int) bool {
			return trainers[i0] >= players[i]
		})
		if idx != len(trainers) {
			ans++
			trainers = trainers[idx+1:]
		}
	}
	return ans
}

func smallestSubarrays(nums []int) []int {
	n := len(nums)
	maxV := make([]int, n)
	or := 0
	for i := n - 1; i >= 0; i-- {
		or |= nums[i]
		maxV[i] = or
	}
	st := segment_tree.InitSegmentTree(nums, func(a, b int) int {
		return a | b
	})
	ans := make([]int, n)
	for i := 0; i < n; i++ {
		idx := sort.Search(n, func(i0 int) bool {
			if i > i0 {
				return false
			}
			return st.Query(i, i0) >= maxV[i]
		})
		ans[i] = idx - i + 1
	}
	return ans
}

func minimumMoney(transactions [][]int) int64 {
	i := -1
	for j := range transactions {
		if transactions[j][0]-transactions[j][1] > 0 {
			i++
			transactions[i], transactions[j] = transactions[j], transactions[i]
		}
	}
	t1, t2 := transactions[:i+1], transactions[i+1:]
	sort.Slice(t1, func(i, j int) bool {
		if t1[i][1] == t1[j][1] {
			return t1[i][0] > t1[j][0]
		}
		return t1[i][1] < t1[j][1]
	})
	sort.Slice(t2, func(i, j int) bool {
		return t2[i][0] > t2[j][0]
	})
	ans := 0
	cur := 0
	for _, v := range transactions {
		if cur < v[0] {
			ans += v[0] - cur
			cur = v[1]
		} else {
			cur -= v[0]
			cur += v[1]
		}
	}
	return int64(ans)
}

// 827 最大人工岛
func largestIsland(grid [][]int) int {
	m, n := len(grid), len(grid[0])
	cur := 1
	cnt := map[int]int{}
	idx := make([][]int, m)
	for i := range idx {
		idx[i] = make([]int, n)
	}
	var dfs func(x, y int)
	dfs = func(x, y int) {
		idx[x][y] = cur
		cnt[cur]++
		for _, d := range dir4 {
			nx, ny := x+d.x, y+d.y
			if nx >= 0 && nx < m && ny >= 0 && ny < n && grid[nx][ny] == 1 && idx[nx][ny] == 0 {
				dfs(nx, ny)
			}
		}
	}
	for i := range grid {
		for j := range grid[i] {
			if grid[i][j] == 1 && idx[i][j] == 0 {
				dfs(i, j)
				cur++
			}
		}
	}
	ans := 1
	for i := range grid {
		for j := range grid[i] {
			if grid[i][j] == 0 {
				sur := map[int]struct{}{}
				for _, d := range dir4 {
					nx, ny := i+d.x, j+d.y
					if nx < m && nx >= 0 && ny < n && ny >= 0 && idx[nx][ny] != 0 {
						sur[idx[nx][ny]] = struct{}{}
					}
				}
				u := 0
				for k := range sur {
					u += cnt[k]
				}
				ans = max(ans, u+1)
			} else {
				ans = max(ans, cnt[idx[i][j]])
			}
		}
	}
	return ans
}

func smallestEvenMultiple(n int) int {
	return LCM(2, n)
}

func longestContinuousSubstring(s string) int {
	ans := 1
	cnt := 1
	for i := 1; i < len(s); i++ {
		if s[i] == s[i-1]+1 {
			cnt++
			ans = max(ans, cnt)
		} else {
			cnt = 1
		}
	}
	return ans
}

func reverseOddLevels(root *TreeNode) *TreeNode {
	data := []int{0}
	queue := []*TreeNode{root}
	for len(queue) > 0 {
		n := len(queue)
		for i := 0; i < n; i++ {
			data = append(data, queue[i].Val)
			if queue[i].Left != nil {
				queue = append(queue, queue[i].Left)
			}
			if queue[i].Right != nil {
				queue = append(queue, queue[i].Right)
			}
		}
		queue = queue[n:]
	}
	cnt := 0
	for i := 1; i < len(data); i <<= 1 {
		if cnt%2 == 1 {
			reverseSlice(data[i : i<<1])
		}
		cnt++
	}
	var ins func(i int) *TreeNode
	ins = func(i int) *TreeNode {
		if i >= len(data) {
			return nil
		}
		return &TreeNode{
			Left:  ins(i * 2),
			Right: ins(i*2 + 1),
			Val:   data[i],
		}
	}
	return ins(1)
}

// 1636 按照频率将数组升序排序
func frequencySort(nums []int) []int {
	cnt := CountInt(nums)
	sort.Slice(nums, func(i, j int) bool {
		if cnt[nums[i]] == cnt[nums[j]] {
			return nums[i] > nums[j]
		}
		return cnt[nums[i]] < cnt[nums[j]]
	})
	return nums
}

// 1640 能否连接形成数组
func canFormArray(arr []int, pieces [][]int) bool {
	m := [101]int{}
	for i := range arr {
		m[arr[i]] = i
	}
	sort.Slice(pieces, func(i, j int) bool {
		return m[pieces[i][0]] < m[pieces[j][0]]
	})
	idx := 0
	for i := range pieces {
		for j := range pieces[i] {
			if pieces[i][j] != arr[idx] {
				return false
			}
			idx++
		}
	}
	return true
}

// 面试题 17.19 消失的两个数字
func missingTwo(nums []int) (ans []int) {
	nums = append(nums, 0, -1, -1)
	for i := range nums {
		for nums[i] != -1 && nums[i] != i {
			nums[i], nums[nums[i]] = nums[nums[i]], nums[i]
		}
	}
	for i := range nums {
		if nums[i] == -1 {
			ans = append(ans, i)
		}
	}
	return
}

// 面试题 17.09 第 k 个数
func getKthMagicNumber(k int) int {
	h := heaps.InitIntMinHeap([]int{1})
	vis := map[int]struct{}{}
	insert := func(x int) {
		if _, ok := vis[x]; !ok {
			vis[x] = struct{}{}
			h.Push(x)
		}
	}
	for i := 1; i < k; i++ {
		top := h.Pop()
		insert(top * 3)
		insert(top * 5)
		insert(top * 7)
	}
	return h.Top()
}

// 面试题 01.08 零矩阵
func setZeroes(matrix [][]int) {
	m, n := len(matrix), len(matrix[0])
	rows, cols := make([]bool, m), make([]bool, n)
	for i := range matrix {
		for j := range matrix[i] {
			if matrix[i][j] == 0 {
				rows[i] = true
				cols[j] = true
			}
		}
	}
	for i := range rows {
		if rows[i] {
			for j := 0; j < n; j++ {
				matrix[i][j] = 0
			}
		}
	}
	for i := range cols {
		if cols[i] {
			for j := 0; j < m; j++ {
				matrix[j][i] = 0
			}
		}
	}
}

// 1694 重新格式化电话号码
func reformatNumber(number string) string {
	sb := strings.Builder{}
	for i := range number {
		if number[i] != '-' && number[i] != ' ' {
			sb.WriteByte(number[i])
		}
	}
	number = sb.String()
	var ns []string
	for k := 0; k < len(number); k += 3 {
		ns = append(ns, number[k:min(k+3, len(number))])
	}
	n := len(ns)
	if n > 1 && len(ns[n-1]) == 1 {
		ns[n-2], ns[n-1] = ns[n-2][:2], ns[n-2][2:]+ns[n-1]
	}
	return strings.Join(ns, "-")
}

// 2423 删除字符使频率相同
func equalFrequency(word string) bool {
	cnt := CountChar(word)
	m := map[int]int{}
	for _, v := range cnt {
		if v != 0 {
			m[v]++
		}
	}
	if len(m) == 1 {
		for k, v := range m {
			if k == 1 || v == 1 {
				return true
			}
		}
	}
	if len(m) == 2 {
		for k, v := range m {
			if k == 1 && v == 1 {
				return true
			}
		}
		ans := MapItems(m)
		if ans[0][1] == 1 && ans[0][0] == ans[1][0]+1 ||
			ans[1][1] == 1 && ans[0][0]+1 == ans[1][0] {
			return true
		}
	}
	return false
}

// 2425 所有数对的异或和
func xorAllNums(nums1 []int, nums2 []int) int {
	if len(nums1)%2 == 0 && len(nums2)%2 == 0 {
		return 0
	}
	xor := func(arr []int) int {
		t := 0
		for _, v := range arr {
			t ^= v
		}
		return t
	}
	if len(nums1)%2 == 0 {
		return xor(nums1)
	}
	if len(nums2)%2 == 0 {
		return xor(nums2)
	}
	return xor(nums1) ^ xor(nums2)
}

// 2426 满足不等式的数对数目
func numberOfPairs1(nums1 []int, nums2 []int, diff int) int64 {
	n := len(nums1)
	stSize := 60010
	off := stSize >> 1
	ans := 0
	st := segment_tree.InitEmptySegmentTree(stSize, func(a, b int) int {
		return a + b
	})
	for i := n - 1; i >= 0; i-- {
		ans += st.Query(nums1[i]-nums2[i]-diff+off, stSize-1)
		st.UpdateFunc(nums1[i]-nums2[i]+off, func(oldVal int) (newVal int) {
			return oldVal + 1
		})
	}
	return int64(ans)
}

// 777 在LR字符串中交换相邻字符
func canTransform(start string, end string) bool {
	n, p, q := len(start), 0, 0
	for p < n && q < n {
		if start[p] == 'X' {
			p++
		} else if end[q] == 'X' {
			q++
		} else {
			if start[p] != end[q] {
				return false
			}
			if start[p] == 'L' && p < q {
				return false
			}
			if start[p] == 'R' && p > q {
				return false
			}
			p++
			q++
		}
	}
	for p < n {
		if start[p] != 'X' {
			return false
		}
		p++
	}
	for q < n {
		if end[q] != 'X' {
			return false
		}
		q++
	}
	return true
}

// 1784 检查二进制字符串字段
func checkOnesSegment(s string) bool {
	return strings.Count(s+"0", "10") <= 1
}

// 921 使括号有效的最少添加
func minAddToMakeValid(s string) int {
	cnt := 0
	ans := 0
	for i := range s {
		if s[i] == '(' {
			cnt++
		} else {
			if cnt == 0 {
				ans++
			} else {
				cnt--
			}
		}
	}
	return ans + cnt
}

// 811 子域名访问计数
func subdomainVisits(cpdomains []string) (ans []string) {
	m := make(map[string]int, len(cpdomains)*len(cpdomains))
	for _, cpd := range cpdomains {
		sp := strings.Split(cpd, " ")
		cnt, _ := strconv.Atoi(sp[0])
		m[sp[1]] += cnt
		for i := range sp[1] {
			if sp[1][i] == '.' {
				m[sp[1][i+1:]] += cnt
			}
		}
	}
	for k, v := range m {
		ans = append(ans, fmt.Sprintf("%d %s", v, k))
	}
	return
}

// 927 三等分
func threeEqualParts(arr []int) []int {
	cnt := 0
	for i := range arr {
		if arr[i] == 1 {
			cnt++
		}
	}
	if cnt == 0 {
		return []int{0, 2}
	}
	if cnt%3 != 0 {
		return []int{-1, -1}
	}
	target := cnt / 3
	cur := 0
	sl := [3][2]int{}
	for i := range arr {
		if arr[i] == 1 {
			if cur%target == 0 {
				sl[cur/target][0] = i
			}
			cur++
			if cur%target == 0 {
				sl[cur/target-1][1] = i
			}
		}
	}
	zeros := len(arr) - (sl[2][1] + 1)
	if sl[0][1]+zeros < sl[1][0] && sl[1][1]+zeros < sl[2][0] {
		if CompareIntSlice(arr[sl[0][0]:sl[0][1]+zeros+1], arr[sl[1][0]:sl[1][1]+zeros+1]) == 0 &&
			CompareIntSlice(arr[sl[2][0]:], arr[sl[1][0]:sl[1][1]+zeros+1]) == 0 {
			return []int{sl[0][1] + zeros, sl[1][1] + zeros + 1}
		}
	}
	return []int{-1, -1}
}

// 994 腐烂的橘子
func orangesRotting(grid [][]int) int {
	m, n := len(grid), len(grid[0])
	dp := make([][]int, m)
	const INF int = 1<<31 - 1
	for i := range dp {
		dp[i] = IntRepeat(INF, n)
	}
	bfs := func(x, y int) {
		queue := [][2]int{{x, y}}
		vis := map[[2]int]struct{}{}
		vis[[2]int{x, y}] = struct{}{}
		level := 0
		for len(queue) > 0 {
			l := len(queue)
			for i := 0; i < l; i++ {
				if dp[queue[i][0]][queue[i][1]] <= level {
					continue
				}
				dp[queue[i][0]][queue[i][1]] = level
				for _, d := range dir4 {
					nx, ny := queue[i][0]+d.x, queue[i][1]+d.y
					if nx < m && nx >= 0 && ny < n && ny >= 0 && grid[nx][ny] == 1 {
						next := [2]int{nx, ny}
						if _, ok := vis[next]; !ok {
							vis[next] = struct{}{}
							queue = append(queue, next)
						}
					}
				}
			}
			level++
			queue = queue[l:]
		}
	}
	for i := range grid {
		for j := range grid[i] {
			if grid[i][j] == 2 {
				bfs(i, j)
			}
		}
	}
	ans := 0
	for i := range grid {
		for j := range grid[i] {
			if grid[i][j] == 1 {
				if dp[i][j] == INF {
					return -1
				}
				ans = max(ans, dp[i][j])
			}
		}
	}
	return ans
}
