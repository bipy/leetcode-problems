package main

import (
	"fmt"
	"testing"
)

func Test_alienOrder(t *testing.T) {
	fmt.Println(alienOrder(StrToStrSlice("[\"wrt\",\"wrf\",\"er\",\"ett\",\"rftt\"]")))
}

func Test_calculateTax(t *testing.T) {
	fmt.Println(calculateTax(StrTo2DIntSlice("[[3,50],[7,10],[12,25]]"), 10))
}

func Test_minPathCost(t *testing.T) {
	fmt.Println(minPathCost(StrTo2DIntSlice("[[5,3],[4,0],[2,1]]"), StrTo2DIntSlice("[[9,8],[1,5],[10,12],[18,6],[2,4],[14,3]]")))
}
