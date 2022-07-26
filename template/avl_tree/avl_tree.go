package avl_tree

type node struct {
	key    int
	val    any
	left   *node
	right  *node
	height int
}

type AVLTree struct {
	root *node
	size int
}

func getNodeHeight(cur *node) int {
	if cur == nil {
		return 0
	}
	return cur.height
}

func (n *node) isBalanced() bool {
	bf := getNodeHeight(n.left) - getNodeHeight(n.right)
	return bf <= 1 && bf >= -1
}

func (n *node) updateHeight() {
	n.height = max(getNodeHeight(n.left), getNodeHeight(n.right)) + 1
}

func llRotation(cur *node) (ptr *node) {
	ptr = cur.left
	cur.left = ptr.right
	ptr.right = cur
	cur.updateHeight()
	ptr.updateHeight()
	return
}

func rrRotation(cur *node) (ptr *node) {
	ptr = cur.right
	cur.right = ptr.left
	ptr.left = cur
	cur.updateHeight()
	ptr.updateHeight()
	return
}

func lrRotation(cur *node) *node {
	cur.left = rrRotation(cur.left)
	return llRotation(cur)
}

func rlRotation(cur *node) *node {
	cur.right = llRotation(cur.right)
	return rrRotation(cur)
}

func insert(cur *node, key int, value any, treeSize *int) *node {
	if cur == nil {
		cur = &node{
			key: key,
			val: value,
		}
		*treeSize++
	} else {
		if key > cur.key {
			cur.right = insert(cur.right, key, value, treeSize)
			if !cur.isBalanced() {
				if cur.right.key > key {
					cur = rlRotation(cur)
				} else {
					cur = rrRotation(cur)
				}
			}
		} else if key < cur.key {
			cur.left = insert(cur.left, key, value, treeSize)
			if !cur.isBalanced() {
				if cur.left.key < key {
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

func find(cur *node, key int) (value any, ok bool) {
	if cur == nil {
		return nil, false
	}
	if cur.key > key {
		return find(cur.left, key)
	}
	if cur.key < key {
		return find(cur.right, key)
	}
	return cur.val, true
}

func (tree *AVLTree) Put(key int, value any) {
	tree.root = insert(tree.root, key, value, &tree.size)
}

func (tree *AVLTree) Find(key int) (value any, ok bool) {
	return find(tree.root, key)
}

func (tree *AVLTree) Len() int {
	return tree.size
}

func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}
