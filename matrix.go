// Copyright 2023 The Mach Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"
	"math/cmplx"
	"math/rand"

	"github.com/pointlander/pagerank"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/num/quat"
	"gonum.org/v1/gonum/stat"
)

const (
	// S is the scaling factor for the softmax
	S = 1.0 - 1e-300
)

const (
	// StateM is the state for the mean
	StateM = iota
	// StateV is the state for the variance
	StateV
	// StateTotal is the total number of states
	StateTotal
)

// Matrix is a matrix
type Matrix struct {
	Cols   int
	Rows   int
	Data   []float64
	States [][]float64
}

// NewMatrix creates a new matrix
func NewMatrix(states, cols, rows int) Matrix {
	m := Matrix{
		Cols: cols,
		Rows: rows,
		Data: make([]float64, 0, cols*rows),
	}
	if states > 0 {
		m.States = make([][]float64, states)
		for i := range m.States {
			m.States[i] = make([]float64, cols*rows)
		}
	}
	return m
}

// NewRandMatrix creates a new random matrix
func NewRandMatrix(rnd *rand.Rand, states, cols, rows int) Matrix {
	m := Matrix{
		Cols: cols,
		Rows: rows,
		Data: make([]float64, 0, cols*rows),
	}
	factor := math.Sqrt(2.0 / float64(cols))
	for i := 0; i < cols*rows; i++ {
		m.Data = append(m.Data, rnd.NormFloat64()*factor)
	}
	if states > 0 {
		m.States = make([][]float64, states)
		for i := range m.States {
			m.States[i] = make([]float64, cols*rows)
		}
	}
	return m
}

// Size is the size of the matrix
func (m Matrix) Size() int {
	return m.Cols * m.Rows
}

// Mul multiplies two matrices
func Mul(m Matrix, n Matrix) Matrix {
	if m.Cols != n.Cols {
		panic(fmt.Errorf("%d != %d", m.Cols, n.Cols))
	}
	columns := m.Cols
	o := Matrix{
		Cols: m.Rows,
		Rows: n.Rows,
		Data: make([]float64, 0, m.Rows*n.Rows),
	}
	lenn, lenm := len(n.Data), len(m.Data)
	for i := 0; i < lenn; i += columns {
		nn := n.Data[i : i+columns]
		for j := 0; j < lenm; j += columns {
			mm := m.Data[j : j+columns]
			o.Data = append(o.Data, dot(mm, nn))
		}
	}
	return o
}

// H element wise multiplies two matrices
func H(m Matrix, n Matrix) Matrix {
	lena, lenb := len(m.Data), len(n.Data)
	if lena%lenb != 0 {
		panic(fmt.Errorf("%d %% %d != 0", lena, lenb))
	}

	o := Matrix{
		Cols: m.Cols,
		Rows: m.Rows,
		Data: make([]float64, 0, m.Cols*m.Rows),
	}
	for i, value := range m.Data {
		o.Data = append(o.Data, value*n.Data[i%lenb])
	}
	return o
}

// Add adds two matrices
func Add(m Matrix, n Matrix) Matrix {
	lena, lenb := len(m.Data), len(n.Data)
	if lena%lenb != 0 {
		panic(fmt.Errorf("%d %% %d != 0", lena, lenb))
	}

	o := Matrix{
		Cols: m.Cols,
		Rows: m.Rows,
		Data: make([]float64, 0, m.Cols*m.Rows),
	}
	for i, value := range m.Data {
		o.Data = append(o.Data, value+n.Data[i%lenb])
	}
	return o
}

// Sub subtracts two matrices
func Sub(m Matrix, n Matrix) Matrix {
	lena, lenb := len(m.Data), len(n.Data)
	if lena%lenb != 0 {
		panic(fmt.Errorf("%d %% %d != 0", lena, lenb))
	}

	o := Matrix{
		Cols: m.Cols,
		Rows: m.Rows,
		Data: make([]float64, 0, m.Cols*m.Rows),
	}
	for i, value := range m.Data {
		o.Data = append(o.Data, value-n.Data[i%lenb])
	}
	return o
}

// Softmax is the softmax of a matrix
func Softmax(m Matrix) Matrix {
	size, width := len(m.Data), m.Cols
	o := Matrix{
		Cols: m.Cols,
		Rows: m.Rows,
		Data: make([]float64, 0, m.Cols*m.Rows),
	}
	max := 0.0
	for _, v := range m.Data {
		if v > max {
			max = v
		}
	}
	values := make([]float64, width)
	for i := 0; i < size; i += width {
		s := max * S
		sum := 0.0
		for j, ax := range m.Data[i : i+width] {
			values[j] = math.Exp(ax - s)
			sum += values[j]
		}
		for _, cx := range values {
			o.Data = append(o.Data, cx/sum)
		}
	}
	return o
}

// Normalize normalizes a matrix to the unit vector
func Normalize(m Matrix) Matrix {
	size, width := len(m.Data), m.Cols
	o := Matrix{
		Cols: m.Cols,
		Rows: m.Rows,
		Data: make([]float64, 0, m.Cols*m.Rows),
	}
	for i := 0; i < size; i += width {
		sum := 0.0
		for _, ax := range m.Data[i : i+width] {
			sum += ax * ax
		}
		length := math.Sqrt(sum)
		if sum == 0 {
			length = 1
		}
		for _, ax := range m.Data[i : i+width] {
			o.Data = append(o.Data, ax/length)
		}
	}
	return o
}

// NormalizeCenter normalizes a matrix using the mean and standard deviation
func NormalizeCenter(m Matrix) Matrix {
	size, width := len(m.Data), m.Cols
	o := Matrix{
		Cols: m.Cols,
		Rows: m.Rows,
		Data: make([]float64, 0, m.Cols*m.Rows),
	}
	sum, sumSquared := 0.0, 0.0
	for i := 0; i < size; i += width {
		for _, ax := range m.Data[i : i+width] {
			sum += ax
			sumSquared += ax * ax
		}
	}
	mean, stddev := 0.0, 0.0
	mean = sum / float64(m.Rows*m.Cols)
	stddev = math.Sqrt(sumSquared/float64(m.Rows*m.Cols) - mean*mean)
	for i := 0; i < size; i += width {
		for _, ax := range m.Data[i : i+width] {
			o.Data = append(o.Data, (ax-mean)/stddev)
		}
	}
	return o
}

// NormalizeCenterPer normalizes a matrix using the mean and standard deviation per vector entry
func NormalizeCenterPer(m Matrix) Matrix {
	size, width := len(m.Data), m.Cols
	o := Matrix{
		Cols: m.Cols,
		Rows: m.Rows,
		Data: make([]float64, 0, m.Cols*m.Rows),
	}
	sum, sumSquared := make([]float64, width), make([]float64, width)
	for i := 0; i < size; i += width {
		for j, ax := range m.Data[i : i+width] {
			sum[j] += ax
			sumSquared[j] += ax * ax
		}
	}
	mean, stddev := make([]float64, width), make([]float64, width)
	for i := 0; i < width; i++ {
		mean[i] = sum[i] / float64(m.Rows)
		stddev[i] = math.Sqrt(sumSquared[i]/float64(m.Rows) - mean[i]*mean[i])
	}
	for i := 0; i < size; i += width {
		for j, ax := range m.Data[i : i+width] {
			o.Data = append(o.Data, (ax-mean[j])/stddev[j])
		}
	}
	return o
}

// Entropy is the entropy of the matrix
func Entropy(m Matrix) Matrix {
	size, width := len(m.Data), m.Cols
	o := Matrix{
		Cols: m.Rows,
		Rows: 1,
		Data: make([]float64, 0, m.Rows),
	}
	for i := 0; i < size; i += width {
		sum := 0.0
		for k := 0; k < width; k++ {
			ax := m.Data[i+k]
			sum += ax * math.Log(ax)
		}
		o.Data = append(o.Data, -sum)
	}
	return o
}

// Neg negates a matrix
func Neg(m Matrix) Matrix {
	o := Matrix{
		Cols: m.Cols,
		Rows: m.Rows,
		Data: make([]float64, 0, m.Cols*m.Rows),
	}
	for _, value := range m.Data {
		o.Data = append(o.Data, -value)
	}
	return o
}

// Sigmoid computes the sigmoid of a matrix
func Sigmoid(m Matrix) Matrix {
	o := Matrix{
		Cols: m.Cols,
		Rows: m.Rows,
		Data: make([]float64, 0, m.Cols*m.Rows),
	}
	for _, value := range m.Data {
		o.Data = append(o.Data, 1/(1+math.Exp(-value)))
	}
	return o
}

// Abs computes the abs of a matrix
func Abs(m Matrix) Matrix {
	o := Matrix{
		Cols: m.Cols,
		Rows: m.Rows,
		Data: make([]float64, 0, m.Cols*m.Rows),
	}
	for _, value := range m.Data {
		o.Data = append(o.Data, math.Abs(value))
	}
	return o
}

// T tramsposes a matrix
func T(m Matrix) Matrix {
	o := Matrix{
		Cols: m.Rows,
		Rows: m.Cols,
		Data: make([]float64, 0, m.Cols*m.Rows),
	}
	for i := 0; i < m.Cols; i++ {
		for j := 0; j < m.Rows; j++ {
			o.Data = append(o.Data, m.Data[j*m.Cols+i])
		}
	}
	return o
}

func softmax(values []float64) {
	max := 0.0
	for _, v := range values {
		if v > max {
			max = v
		}
	}
	s := max * S
	sum := 0.0
	for j, value := range values {
		values[j] = math.Exp(value - s)
		sum += values[j]
	}
	for j, value := range values {
		values[j] = value / sum
	}
}

// PageRank computes the page rank of the adjacency matrix
func PageRank(m Matrix) Matrix {
	o := Matrix{
		Cols: m.Cols,
		Rows: 1,
		Data: make([]float64, m.Cols),
	}
	graph := pagerank.NewGraph64()
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			graph.Link(uint64(i), uint64(j), m.Data[i*m.Cols+j])
		}
	}
	graph.Rank(0.85, 0.000001, func(node uint64, rank float64) {
		o.Data[node] = rank
	})
	return o
}

// PCA computes the principal component analysis of the matrix
func PCA(m Matrix) (Matrix, *mat.Dense) {
	o := Matrix{
		Data: make([]float64, 0, m.Cols*m.Rows),
	}
	data := mat.NewDense(m.Rows, m.Cols, m.Data)

	var pc stat.PC
	ok := pc.PrincipalComponents(data, nil)
	if !ok {
		panic("failed to compute principal components")
	}

	var projection mat.Dense
	var vector mat.Dense
	pc.VectorsTo(&vector)
	projection.Mul(data, &vector)
	rows, cols := projection.Dims()
	o.Cols = cols
	o.Rows = rows
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			o.Data = append(o.Data, projection.At(i, j))
		}
	}
	return o, &vector
}

// SlowSelfEntropy computes the slowself entropy of Q, K, V
func SlowSelfEntropy(Q, K, V Matrix) []float64 {
	A := Mul(Q, K)
	R := PageRank(A)
	A = H(A, R)
	E := Entropy(Softmax(T(Mul(Softmax(A), T(V)))))
	results := make([]float64, 0, E.Rows)
	for i := 0; i < E.Cols; i++ {
		results = append(results, E.Data[i])
	}
	return results
}

// SelfEntropy computes the self entropy of Q, K, V
func SelfEntropy(Q, K, V Matrix) []float64 {
	entropies, values, results := make([]float64, V.Cols), make([]float64, K.Rows), make([]float64, 0, K.Rows)
	V = T(V)
	for i := 0; i < K.Rows; i++ {
		K := K.Data[i*K.Cols : (i+1)*K.Cols]
		for j := 0; j < Q.Rows; j++ {
			Q := Q.Data[j*Q.Cols : (j+1)*Q.Cols]
			values[j] = dot(K, Q)
		}
		softmax(values)

		for j := 0; j < V.Rows; j++ {
			V := V.Data[j*V.Cols : (j+1)*V.Cols]
			entropies[j] = dot(values, V)
		}
		softmax(entropies)

		entropy := 0.0
		for _, e := range entropies {
			entropy += e * math.Log(e)
		}
		results = append(results, entropy)
	}
	return results
}

// AppendOne appends m and n
func Append(m, n Matrix) Matrix {
	if m.Rows != n.Rows {
		panic(fmt.Errorf("%d != %d", m.Rows, n.Rows))
	}
	o := Matrix{
		Cols: m.Cols + n.Cols,
		Rows: m.Rows,
		Data: make([]float64, 0, (m.Cols+n.Cols)*m.Rows),
	}
	for i := 0; i < m.Rows; i++ {
		o.Data = append(o.Data, m.Data[i*m.Cols:i*m.Cols+m.Cols]...)
		o.Data = append(o.Data, n.Data[i*n.Cols:i*n.Cols+n.Cols]...)
	}
	return o
}

// SelfAttention computes the self attention of Q, K, V
func SelfAttention(Q, K, V Matrix) Matrix {
	o := Matrix{
		Cols: Q.Cols,
		Rows: Q.Rows,
		Data: make([]float64, 0, Q.Cols*Q.Rows),
	}
	outputs, values := make([]float64, V.Cols), make([]float64, K.Rows)
	V = T(V)
	for i := 0; i < K.Rows; i++ {
		K := K.Data[i*K.Cols : (i+1)*K.Cols]
		for j := 0; j < Q.Rows; j++ {
			Q := Q.Data[j*Q.Cols : (j+1)*Q.Cols]
			values[j] = dot(K, Q)
		}
		softmax(values)

		for j := 0; j < V.Rows; j++ {
			V := V.Data[j*V.Cols : (j+1)*V.Cols]
			outputs[j] = dot(values, V)
		}
		o.Data = append(o.Data, outputs...)
	}
	return o
}

func complexDot(X, Y []complex128) complex128 {
	var sum complex128
	for i, x := range X {
		sum += x * Y[i]
	}
	return sum
}

// ComplexMatrix is a complex matrix
type ComplexMatrix struct {
	Cols   int
	Rows   int
	Data   []complex128
	States [][]complex128
}

// NewComplexMatrix creates a new complex matrix
func NewComplexMatrix(states, cols, rows int) ComplexMatrix {
	m := ComplexMatrix{
		Cols: cols,
		Rows: rows,
		Data: make([]complex128, 0, cols*rows),
	}
	if states > 0 {
		m.States = make([][]complex128, states)
		for i := range m.States {
			m.States[i] = make([]complex128, cols*rows)
		}
	}
	return m
}

// NewComplexIdentityMatrix creates a new complex matrix
func NewComplexIdentityMatrix(states, cols, rows int) ComplexMatrix {
	m := ComplexMatrix{
		Cols: cols,
		Rows: rows,
		Data: make([]complex128, 0, cols*rows),
	}
	for i := 0; i < cols; i++ {
		for j := 0; j < rows; j++ {
			if i == j {
				m.Data = append(m.Data, 1)
			} else {
				m.Data = append(m.Data, 0)
			}
		}
	}
	if states > 0 {
		m.States = make([][]complex128, states)
		for i := range m.States {
			m.States[i] = make([]complex128, cols*rows)
		}
	}
	return m
}

// ComplexMul multiplies two complex matrices
func ComplexMul(m ComplexMatrix, n ComplexMatrix) ComplexMatrix {
	if m.Cols != n.Cols {
		panic(fmt.Errorf("%d != %d", m.Cols, n.Cols))
	}
	columns := m.Cols
	o := ComplexMatrix{
		Cols: m.Rows,
		Rows: n.Rows,
		Data: make([]complex128, 0, m.Rows*n.Rows),
	}
	lenn, lenm := len(n.Data), len(m.Data)
	for i := 0; i < lenn; i += columns {
		nn := n.Data[i : i+columns]
		for j := 0; j < lenm; j += columns {
			mm := m.Data[j : j+columns]
			o.Data = append(o.Data, complexDot(mm, nn))
		}
	}
	return o
}

// ComplexT tramsposes a complex matrix
func ComplexT(m ComplexMatrix) ComplexMatrix {
	o := ComplexMatrix{
		Cols: m.Rows,
		Rows: m.Cols,
		Data: make([]complex128, 0, m.Cols*m.Rows),
	}
	for i := 0; i < m.Cols; i++ {
		for j := 0; j < m.Rows; j++ {
			o.Data = append(o.Data, m.Data[j*m.Cols+i])
		}
	}
	return o
}

// ComplexConj computes the complex conjugate of a matrix
func ComplexConj(m ComplexMatrix) ComplexMatrix {
	o := ComplexMatrix{
		Cols: m.Cols,
		Rows: m.Rows,
		Data: make([]complex128, 0, m.Cols*m.Rows),
	}
	for _, value := range m.Data {
		o.Data = append(o.Data, cmplx.Conj(value))
	}
	return o
}

// ComplexSub subtracts two complex matrices
func ComplexSub(m ComplexMatrix, n ComplexMatrix) ComplexMatrix {
	lena, lenb := len(m.Data), len(n.Data)
	if lena%lenb != 0 {
		panic(fmt.Errorf("%d %% %d != 0", lena, lenb))
	}

	o := ComplexMatrix{
		Cols: m.Cols,
		Rows: m.Rows,
		Data: make([]complex128, 0, m.Cols*m.Rows),
	}
	for i, value := range m.Data {
		o.Data = append(o.Data, value-n.Data[i%lenb])
	}
	return o
}

func qDot(X, Y []quat.Number) quat.Number {
	var sum quat.Number
	for i, x := range X {
		sum = quat.Add(sum, quat.Mul(x, Y[i]))
	}
	return sum
}

// QMatrix is a quaternion matrix
type QMatrix struct {
	Cols   int
	Rows   int
	Data   []quat.Number
	States [][]quat.Number
}

// NewQMatrix creates a new quaternion matrix
func NewQMatrix(states, cols, rows int) QMatrix {
	m := QMatrix{
		Cols: cols,
		Rows: rows,
		Data: make([]quat.Number, 0, cols*rows),
	}
	if states > 0 {
		m.States = make([][]quat.Number, states)
		for i := range m.States {
			m.States[i] = make([]quat.Number, cols*rows)
		}
	}
	return m
}

// NewQIdentityMatrix creates a new quaternion matrix
func NewQIdentityMatrix(states, cols, rows int) QMatrix {
	m := QMatrix{
		Cols: cols,
		Rows: rows,
		Data: make([]quat.Number, 0, cols*rows),
	}
	for i := 0; i < cols; i++ {
		for j := 0; j < rows; j++ {
			if i == j {
				m.Data = append(m.Data, quat.Number{Real: 1})
			} else {
				m.Data = append(m.Data, quat.Number{Real: 0})
			}
		}
	}
	if states > 0 {
		m.States = make([][]quat.Number, states)
		for i := range m.States {
			m.States[i] = make([]quat.Number, cols*rows)
		}
	}
	return m
}

// QMul multiplies two quaternion matrices
func QMul(m QMatrix, n QMatrix) QMatrix {
	if m.Cols != n.Cols {
		panic(fmt.Errorf("%d != %d", m.Cols, n.Cols))
	}
	columns := m.Cols
	o := QMatrix{
		Cols: m.Rows,
		Rows: n.Rows,
		Data: make([]quat.Number, 0, m.Rows*n.Rows),
	}
	lenn, lenm := len(n.Data), len(m.Data)
	for i := 0; i < lenn; i += columns {
		nn := n.Data[i : i+columns]
		for j := 0; j < lenm; j += columns {
			mm := m.Data[j : j+columns]
			o.Data = append(o.Data, qDot(mm, nn))
		}
	}
	return o
}

// QT tramsposes a quaternion matrix
func QT(m QMatrix) QMatrix {
	o := QMatrix{
		Cols: m.Rows,
		Rows: m.Cols,
		Data: make([]quat.Number, 0, m.Cols*m.Rows),
	}
	for i := 0; i < m.Cols; i++ {
		for j := 0; j < m.Rows; j++ {
			o.Data = append(o.Data, m.Data[j*m.Cols+i])
		}
	}
	return o
}

// QConj computes the quaternion conjugate of a matrix
func QConj(m QMatrix) QMatrix {
	o := QMatrix{
		Cols: m.Rows,
		Rows: m.Cols,
		Data: make([]quat.Number, 0, m.Cols*m.Rows),
	}
	for _, value := range m.Data {
		o.Data = append(o.Data, quat.Conj(value))
	}
	return o
}

// QSub subtracts two quaternion matrices
func QSub(m QMatrix, n QMatrix) QMatrix {
	lena, lenb := len(m.Data), len(n.Data)
	if lena%lenb != 0 {
		panic(fmt.Errorf("%d %% %d != 0", lena, lenb))
	}

	o := QMatrix{
		Cols: m.Cols,
		Rows: m.Rows,
		Data: make([]quat.Number, 0, m.Cols*m.Rows),
	}
	for i, value := range m.Data {
		o.Data = append(o.Data, quat.Sub(value, n.Data[i%lenb]))
	}
	return o
}
