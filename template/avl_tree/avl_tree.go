package avl_tree

type avlNode struct {
	key    interface{}
	val    interface{}
	left   *avlNode
	right  *avlNode
	height int
}

type AVLTree struct {
	root *avlNode
	size int
	less func(a, b interface{}) bool
}

func InitAVLTree(less func(i, j interface{}) bool) *AVLTree {
	return &AVLTree{less: less}
}

func getNodeHeight(cur *avlNode) int {
	if cur == nil {
		return 0
	}
	return cur.height
}

func (n *avlNode) isBalanced() bool {
	bf := getNodeHeight(n.left) - getNodeHeight(n.right)
	return bf <= 1 && bf >= -1
}

func (n *avlNode) updateHeight() {
	n.height = max(getNodeHeight(n.left), getNodeHeight(n.right)) + 1
}

func llRotation(cur *avlNode) (ptr *avlNode) {
	ptr = cur.left
	cur.left = ptr.right
	ptr.right = cur
	cur.updateHeight()
	ptr.updateHeight()
	return
}

func rrRotation(cur *avlNode) (ptr *avlNode) {
	ptr = cur.right
	cur.right = ptr.left
	ptr.left = cur
	cur.updateHeight()
	ptr.updateHeight()
	return
}

func lrRotation(cur *avlNode) *avlNode {
	cur.left = rrRotation(cur.left)
	return llRotation(cur)
}

func rlRotation(cur *avlNode) *avlNode {
	cur.right = llRotation(cur.right)
	return rrRotation(cur)
}

func (t AVLTree) insert(cur *avlNode, key interface{}, value interface{}, treeSize *int) *avlNode {
	if cur == nil {
		cur = &avlNode{
			key: key,
			val: value,
		}
		*treeSize++
	} else {
		if t.less(cur.key, key) {
			cur.right = t.insert(cur.right, key, value, treeSize)
			if !cur.isBalanced() {
				if t.less(key, cur.right.key) {
					cur = rlRotation(cur)
				} else {
					cur = rrRotation(cur)
				}
			}
		} else if t.less(key, cur.key) {
			cur.left = t.insert(cur.left, key, value, treeSize)
			if !cur.isBalanced() {
				if t.less(cur.left.key, key) {
					cur = lrRotation(cur)
				} else {
					cur = llRotation(cur)
				}
			}
		} else {
			cur.val = value
		}
	}
	cur.updateHeight()
	return cur
}

func (t AVLTree) find(cur *avlNode, key interface{}) (value interface{}, ok bool) {
	if cur == nil {
		return nil, false
	}
	if t.less(key, cur.key) {
		return t.find(cur.left, key)
	}
	if t.less(cur.key, key) {
		return t.find(cur.right, key)
	}
	return cur.val, true
}

func (t *AVLTree) Put(key interface{}, value interface{}) {
	t.root = t.insert(t.root, key, value, &t.size)
}

func (t AVLTree) Get(key interface{}) (value interface{}, ok bool) {
	return t.find(t.root, key)
}

func (t AVLTree) Size() int {
	return t.size
}

func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}
