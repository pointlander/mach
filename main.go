// Copyright 2023 The Mach Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/color/palette"
	drw "image/draw"
	"image/gif"
	"math"
	"math/cmplx"
	"math/rand"
	"os"
	"time"

	gl "github.com/fogleman/fauxgl"
	"github.com/mjibson/go-dsp/dsputils"
	"github.com/mjibson/go-dsp/fft"
	"github.com/nfnt/resize"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
	"gonum.org/v1/plot/vg/vgimg"
)

var (
	// FlagMin minimize entropy
	FlagMin = flag.Bool("min", false, "minimize entropy")
	// FlagMax maximize entropy
	FlagMax = flag.Bool("max", false, "maximize entropy")
	// FlagConstance constant entropy
	FlagConstance = flag.Bool("const", false, "constant entropy")
	// FlagN is the number of particles
	FlagN = flag.Int("n", 0, "number of particles")
	// FlagFFT is the fft mode
	FlagFFT = flag.Bool("fft", false, "fft mode")
	// FlagUnitary is the unitary mode
	FlagUnitary = flag.Bool("unitary", false, "unitary mode")
	// FlagQuaternion is the quaternion mode
	FlagQuaternion = flag.Bool("quaternion", false, "quaternion mode")
)

// Particle is a particle
type Particle struct {
	X, Y, T float64
}

// Optimizer is an optimizer
type Optimizer interface {
	Optimize(current, next float64) bool
}

// MinOptimizer is a minimization optimizer
type MinOptimizer struct {
	Min float64
}

// Optimize is a minimization optimizer
func (o *MinOptimizer) Optimize(current, next float64) bool {
	if next < o.Min {
		o.Min = next
		return true
	}
	return false
}

// MaxOptimizer is a maximization optimizer
type MaxOptimizer struct {
	Max float64
}

// Optimize is a maximization optimizer
func (o *MaxOptimizer) Optimize(current, next float64) bool {
	if next > o.Max {
		o.Max = next
		return true
	}
	return false
}

// ConstanceOptimizer is a constance optimizer
type ConstanceOptimizer struct {
	Min float64
}

// Optimize is a constance optimizer
func (o *ConstanceOptimizer) Optimize(current, next float64) bool {
	if math.Abs(next-current) < o.Min {
		o.Min = math.Abs(next - current)
		return true
	}
	return false
}

const sets = 8

var n = 4

// GetEntropy is a function to get entropy
type GetEntropy func([]Particle) []float64

// SelfAttentionGetEntropy usese self attention to get entropy
func SelfAttentionGetEntropy(particles []Particle) []float64 {
	distances := NewMatrix(0, len(particles), len(particles))
	for _, a := range particles {
		for _, b := range particles {
			d := math.Sqrt(math.Pow(a.X-b.X, 2) + math.Pow(a.Y-b.Y, 2) + math.Pow(a.T-b.T, 2))
			distances.Data = append(distances.Data, d)
		}
	}
	units := Normalize(distances)
	embedding := SelfEntropy(units, units, units)
	return embedding
}

// FFTGetEntropy uses FFT to get entropy
func FFTGetEntropy(particles []Particle) []float64 {
	const size = 32
	verse := dsputils.MakeEmptyMatrix([]int{sets, size, size})
	for i := 0; i < sets; i++ {
		for j := 0; j < 4; j++ {
			x := int(particles[i*4+j].X) % size
			if x < 0 {
				x += size
			}
			y := int(particles[i*4+j].Y) % size
			if y < 0 {
				y += size
			}
			dim := []int{i, x, y}
			verse.SetValue(1, dim)
		}
	}
	verse = fft.FFTN(verse)
	p, index := make([]float64, sets*size*size), 0
	for i := 0; i < sets; i++ {
		for j := 0; j < size; j++ {
			for k := 0; k < size; k++ {
				r := cmplx.Abs(verse.Value([]int{i, j, k}))
				p[index] = r * r
				index++
			}
		}
	}
	sum := 0.0
	for _, value := range p {
		sum += value
	}
	h := 0.0
	for _, value := range p {
		value /= sum
		value += .5
		h += value * math.Log(value)
	}
	return []float64{h}
}

// UnitaryMetric is the unitary metrix
func UnitaryMetric(particles []Particle) []float64 {
	particles = particles[len(particles)-n:]
	distances := NewComplexMatrix(0, len(particles), len(particles))
	for _, a := range particles {
		for _, b := range particles {
			d := complex(b.X-a.X, b.Y-a.Y)
			distances.Data = append(distances.Data, d)
		}
	}
	metric := ComplexSub(ComplexMul(distances, ComplexConj(distances)), NewComplexIdentityMatrix(0, len(particles), len(particles)))
	total := 0.0
	for _, value := range metric.Data {
		total += cmplx.Abs(value)
	}
	return []float64{total}
}

const (
	scale  = 4    // optional supersampling
	width  = 1600 // output width in pixels
	height = 1600 // output height in pixels
	fovy   = 30   // vertical field of view in degrees
	near   = 1    // near clipping plane
	far    = 100  // far clipping plane
)

var (
	eye         = gl.V(3*4, 3*4, 1.5*4)          // camera position
	center      = gl.V(0, 0, 0)                  // view center position
	up          = gl.V(0, 0, 1)                  // up vector
	light       = gl.V(0.75, 0.5, 1).Normalize() // light direction
	objectColor = gl.HexColor("#468966")         // object color
	background  = gl.HexColor("#FFF8E3")         // background color
)

// QuaternionMode is the quaternion mode
func QuaternionMode() {
	mesh := gl.NewEmptyMesh()
	for i := 0; i < 1500; i++ {
		var x, y, z float64
		for {
			x = rand.Float64()*2 - 1
			y = rand.Float64()*2 - 1
			z = rand.Float64()*2 - 1
			if x*x+y*y+z*z < 1 {
				break
			}
		}
		p := gl.Vector{x, y, z}.MulScalar(4)
		s := gl.V(0.2, 0.2, 0.2)
		u := gl.RandomUnitVector()
		a := rand.Float64() * 2 * math.Pi
		c := gl.NewCube()
		c.Transform(gl.Orient(p, s, u, a))
		mesh.Add(c)
	}

	// create a rendering context
	context := gl.NewContext(width*scale, height*scale)
	context.ClearColorBufferWith(gl.Black)

	// create transformation matrix and light direction
	aspect := float64(width) / float64(height)
	matrix := gl.LookAt(eye, center, up).Perspective(fovy, aspect, near, far)

	// render
	shader := gl.NewPhongShader(matrix, light, eye)
	shader.ObjectColor = objectColor
	context.Shader = shader
	start := time.Now()
	context.DrawMesh(mesh)
	fmt.Println(time.Since(start))

	// downsample image for antialiasing
	image := context.Image()
	image = resize.Resize(width, height, image, resize.Bilinear)

	// save image
	gl.SavePNG("out.png", image)
}

func main() {
	rng := rand.New(rand.NewSource(1))
	flag.Parse()

	if *FlagQuaternion {
		QuaternionMode()
		return
	}

	particles := []Particle{
		{X: 0, Y: 0},
		{X: 128, Y: 0},
		{X: 0, Y: 128},
		{X: 64, Y: 64},
	}
	if *FlagFFT {
		particles = []Particle{
			{X: 0, Y: 0},
			{X: 8, Y: 0},
			{X: 0, Y: 8},
			{X: 4, Y: 4},
		}
	}
	if *FlagN > 0 {
		n = *FlagN
		particles = make([]Particle, n)
		for i := 0; i < n; i++ {
			particles[i] = Particle{X: rng.Float64() * 128, Y: rng.Float64() * 128}
		}
	}
	for i := 1; i < sets; i++ {
		for _, particle := range particles[:n] {
			particle.X += rng.NormFloat64()
			particle.Y += rng.NormFloat64()
			particle.T = float64(i)
			particles = append(particles, particle)
		}
	}

	var getEntropy GetEntropy = SelfAttentionGetEntropy
	if *FlagFFT {
		getEntropy = FFTGetEntropy
	} else if *FlagUnitary {
		getEntropy = UnitaryMetric
	}
	const epochs = 256
	images := make([]*image.Paletted, epochs)
	for s := 0; s < epochs; s++ {
		fmt.Println("epcoh:", s)

		var optimizer Optimizer = &MinOptimizer{Min: math.MaxFloat64}
		if *FlagMax {
			optimizer = &MaxOptimizer{Max: -math.MaxFloat64}
		} else if *FlagConstance {
			optimizer = &ConstanceOptimizer{Min: math.MaxFloat64}
		}

		sum, avg := make([]float64, n), make([]float64, n)
		for i := 1; i < sets; i++ {
			for j := 0; j < n; j++ {
				a := particles[(i-1)*n+j]
				b := particles[i*n+j]
				d := math.Sqrt(math.Pow(a.X-b.X, 2) + math.Pow(a.Y-b.Y, 2))
				sum[j] += d
			}
		}
		for i := 0; i < n; i++ {
			avg[i] = sum[i] / float64(sets-1)
		}
		points, length := make(plotter.XYs, 0, len(particles)), len(particles)

		current := 0.0
		entropy := getEntropy(particles)
		for _, value := range entropy {
			current += -value
		}

		saved, best := make([]Particle, n), make([]Particle, n)
		index := 0
		for i := length - n; i < length; i++ {
			saved[index] = particles[i]
			best[index] = particles[i]
			index++
		}
		for s := 0; s < 128; s++ {
			index := 0
			for i := length - n; i < length; i++ {
				particles[i].X += rng.NormFloat64() * avg[index]
				particles[i].Y += rng.NormFloat64() * avg[index]
				index++
			}
			entropy := getEntropy(particles)
			index, sum := 0, 0.0
			for _, value := range entropy {
				sum += -value
			}
			//fmt.Println(s, "entropy:", sum)
			if optimizer.Optimize(current, sum) {
				index := 0
				for i := length - n; i < length; i++ {
					best[index] = particles[i]
					index++
				}
			}
			index = 0
			for i := length - n; i < length; i++ {
				particles[i] = saved[index]
				index++
			}
		}

		index = 0
		for i := length - n; i < length; i++ {
			particles[i] = best[index]
			points = append(points, plotter.XY{X: best[index].X, Y: best[index].Y})
			index++
		}
		for i := length - n; i < length; i++ {
			particle := particles[i]
			particle.T++
			particles = append(particles, particle)
		}
		if len(particles) > n*sets {
			next := make([]Particle, n*sets)
			copy(next, particles[len(particles)-n*sets:])
			particles = next
		}
		p := plot.New()

		p.Title.Text = "x vs y"
		p.X.Label.Text = "x"
		p.Y.Label.Text = "y"

		scatter, err := plotter.NewScatter(points)
		if err != nil {
			panic(err)
		}
		scatter.GlyphStyle.Radius = vg.Length(3)
		scatter.GlyphStyle.Shape = draw.CircleGlyph{}
		p.Add(scatter)

		c := vgimg.New(8*vg.Inch, 8*vg.Inch)
		p.Draw(draw.New(c))

		opts := gif.Options{
			NumColors: 256,
			Drawer:    drw.FloydSteinberg,
		}
		bounds := c.Image().Bounds()

		paletted := image.NewPaletted(bounds, palette.Plan9[:opts.NumColors])
		if opts.Quantizer != nil {
			paletted.Palette = opts.Quantizer.Quantize(make(color.Palette, 0, opts.NumColors), c.Image())
		}
		opts.Drawer.Draw(paletted, bounds, c.Image(), image.Point{})

		images[s] = paletted
	}

	animation := &gif.GIF{}
	for _, paletted := range images {
		animation.Image = append(animation.Image, paletted)
		animation.Delay = append(animation.Delay, 0)
	}

	filename := "min.gif"
	if *FlagMin {
		filename = "min.gif"
	} else if *FlagMax {
		filename = "max.gif"
	} else if *FlagConstance {
		filename = "const.gif"
	}
	f, _ := os.OpenFile(filename, os.O_WRONLY|os.O_CREATE, 0600)
	defer f.Close()
	gif.EncodeAll(f, animation)
}
