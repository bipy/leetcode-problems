package template

type avlNode struct {
	key    int
	val    any
	left   *avlNode
	right  *avlNode
	height int
}

type AVLTree struct {
	root *avlNode
	size int
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

func insert(cur *avlNode, key int, value any, treeSize *int) *avlNode {
	if cur == nil {
		cur = &avlNode{
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

func find(cur *avlNode, key int) (value any, ok bool) {
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
