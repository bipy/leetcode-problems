package skip_list

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestSkipList(t *testing.T) {
	s := InitSkipList(IntComparator)
	for i := 0; i < 10000; i++ {
		s.Insert(i, struct{}{})
	}
	_, ok := s.Find(3333)
	assert.True(t, ok)
}
