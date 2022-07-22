package leetcode

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestRemoveLastOne(t *testing.T) {
	assert.Equal(t, uint(0xfe), RemoveLastOne(0xff))
}
