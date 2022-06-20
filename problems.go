package leetcode

import (
	"fmt"
	"github.com/emirpasic/gods/trees/redblacktree"
	"math"
	"math/rand"
	"net"
	"sort"
	"strconv"
	"strings"
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
	cm := &cntMap{}
	n := len(s)
	ans := int(n)
	cm.add(s[0])
	for i := 2; i <= n; i++ {
		if i%2 == 0 {
			cm.add(s[i-1])
			ans += cm.len
			for j := i; j < n; j++ {
				cm.add(s[j])
				cm.remove(s[j-i])
				ans += cm.len
			}
		} else {
			cm.add(s[n-i])
			ans += cm.len
			for j := n - i - 1; j >= 0; j-- {
				cm.add(s[j])
				cm.remove(s[j+i])
				ans += cm.len
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

func getDistSq(dx, dy int) int {
	return dx*dx + dy*dy
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

func Constructor() CountIntervals {
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
