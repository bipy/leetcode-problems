package leetcode

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestStr2Bytes(t *testing.T) {
	assert.Equal(t, []byte("hello world!"), Str2Bytes("hello world!"))
}

func TestBytes2Str(t *testing.T) {
	assert.Equal(t, "hello world!", Bytes2Str([]byte("hello world!")))
}
