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
	"sort"
	"time"

	gl "github.com/fogleman/fauxgl"
	"github.com/mjibson/go-dsp/dsputils"
	"github.com/mjibson/go-dsp/fft"
	"github.com/nfnt/resize"
	"gonum.org/v1/gonum/num/quat"
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
	// FlagConstant constant entropy
	FlagConstant = flag.Bool("const", false, "constant entropy")
	// FlagAverage is the average mode
	FlagAverage = flag.Bool("average", false, "average mode")
	// FlagN is the number of particles
	FlagN = flag.Int("n", 0, "number of particles")
	// Flag3D is 3d plotting mode
	Flag3D = flag.Bool("3d", false, "3d plotting mode")
	// FlagFFT is the fft mode
	FlagFFT = flag.Bool("fft", false, "fft mode")
	// FlagUnitary is the unitary mode
	FlagUnitary = flag.Bool("unitary", false, "unitary mode")
	// FlagQuaternion is the quaternion mode
	FlagQuaternion = flag.Bool("quaternion", false, "quaternion mode")
)

// Particle is a particle
type Particle struct {
	I          int
	X, Y, Z, T float64
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

// ConstantOptimizer is a constant optimizer
type ConstantOptimizer struct {
	Min float64
}

// Optimize is a constant optimizer
func (o *ConstantOptimizer) Optimize(current, next float64) bool {
	if math.Abs(next-current) < o.Min {
		o.Min = math.Abs(next - current)
		return true
	}
	return false
}

const (
	sets    = 8
	epochs  = 256
	window  = 9
	Samples = 128
)

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
	for key, value := range embedding {
		embedding[key] = -value
	}
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
	return []float64{-h}
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
	metric := ComplexSub(ComplexMul(distances, ComplexConj(ComplexT(distances))), NewComplexIdentityMatrix(0, len(particles), len(particles)))
	total := 0.0
	for _, value := range metric.Data {
		total += cmplx.Abs(value)
	}
	return []float64{total}
}

// QuaternionUnitaryMetric is the quaternion unitary metrix
func QuaternionUnitaryMetric(particles []Particle) []float64 {
	particles = particles[len(particles)-n:]
	distances := NewQMatrix(0, len(particles), len(particles))
	for _, a := range particles {
		for _, b := range particles {
			d := quat.Number{
				Real: b.T - a.T,
				Imag: b.X - a.X,
				Jmag: b.Y - a.Y,
				Kmag: b.Z - a.Z,
			}
			distances.Data = append(distances.Data, d)
		}
	}
	metric := QSub(QMul(distances, QConj(QT(distances))), NewQIdentityMatrix(0, len(particles), len(particles)))

	totals := make([]float64, 0, len(particles))
	for i := range particles {
		total := quat.Number{}
		for _, value := range metric.Data[i*len(particles) : (i+1)*len(particles)] {
			total = quat.Add(total, value)
		}
		totals = append(totals, quat.Abs(total))
	}
	return totals
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
func QuaternionMode(rng *rand.Rand) {
	particles := []Particle{
		{X: 0, Y: 0, Z: 0},
		{X: 1, Y: 0, Z: 0},
		{X: 0, Y: 1, Z: 0},
		{X: .5, Y: .5, Z: 0},
	}
	if *FlagN > 0 {
		n = *FlagN
		particles = make([]Particle, n)
		for i := 0; i < n; i++ {
			particles[i] = Particle{
				X: rng.NormFloat64() * 0.01,
				Y: rng.NormFloat64() * 0.01,
				Z: rng.NormFloat64() * 0.01,
				T: rng.NormFloat64() * 0.01,
			}
		}
	} else {
		for i := 1; i < sets; i++ {
			for _, particle := range particles[:n] {
				particle.X += rng.NormFloat64() * 0.01
				particle.Y += rng.NormFloat64() * 0.01
				particle.Z += rng.NormFloat64() * 0.01
				particle.T += rng.NormFloat64() * 0.01
				particles = append(particles, particle)
			}
		}
	}

	var getEntropy GetEntropy = QuaternionUnitaryMetric
	for s := 0; s < epochs; s++ {
		fmt.Println("epcoh:", s)

		var optimizer Optimizer = &MinOptimizer{Min: math.MaxFloat64}
		if *FlagMax {
			optimizer = &MaxOptimizer{Max: -math.MaxFloat64}
		} else if *FlagConstant {
			optimizer = &ConstantOptimizer{Min: math.MaxFloat64}
		}

		length := len(particles)

		current := 0.0
		currents := getEntropy(particles)
		for _, value := range currents {
			current += value
		}

		saved := make([]Particle, n)
		index := 0
		for i := length - n; i < length; i++ {
			saved[index] = particles[i]
			index++
		}
		type Sample struct {
			Sample  []Particle
			Entropy []float64
			Cost    float64
		}
		samples := make([]Sample, Samples)
		for i := range samples {
			samples[i].Sample = make([]Particle, n)
		}
		gtlt := make([]float64, Samples)
		for s := 0; s < Samples; s++ {
			index := 0
			for i := length - n; i < length; i++ {
				particles[i].T += rng.NormFloat64() * 0.01
				particles[i].X += rng.NormFloat64() * 0.01
				particles[i].Y += rng.NormFloat64() * 0.01
				particles[i].Z += rng.NormFloat64() * 0.01
				index++
			}
			entropy := getEntropy(particles)
			index, sum := 0, 0.0
			for i, value := range entropy {
				if value > currents[i] {
					gtlt[s]++
				} else {
					gtlt[s]--
				}
				sum += value
			}
			//fmt.Println(s, "current:", current, "entropy:", sum)
			index = 0
			for i := length - n; i < length; i++ {
				samples[s].Sample[index] = particles[i]
				index++
			}
			samples[s].Entropy = entropy
			samples[s].Cost = sum
			index = 0
			for i := length - n; i < length; i++ {
				particles[i] = saved[index]
				index++
			}
		}

		if s == 0 || s == epochs-1 {
			d := make(plotter.Values, 0, 8)
			for _, sample := range samples {
				d = append(d, sample.Cost)
			}
			p := plot.New()
			p.Title.Text = "particle distribution"

			histogram, err := plotter.NewHist(d, 10)
			if err != nil {
				panic(err)
			}
			p.Add(histogram)

			err = p.Save(8*vg.Inch, 8*vg.Inch, fmt.Sprintf("particle_distribution_%d.png", s))
			if err != nil {
				panic(err)
			}

			d = make(plotter.Values, 0, 8)
			for _, sample := range gtlt {
				d = append(d, sample)
			}
			p = plot.New()
			p.Title.Text = "gtlt distribution"

			histogram, err = plotter.NewHist(d, 10)
			if err != nil {
				panic(err)
			}
			p.Add(histogram)

			err = p.Save(8*vg.Inch, 8*vg.Inch, fmt.Sprintf("gtlt_distribution_%d.png", s))
			if err != nil {
				panic(err)
			}
		}

		best := []Particle{}
		if *FlagAverage {
			sort.Slice(samples, func(i, j int) bool {
				return samples[i].Cost < samples[j].Cost
			})
			min, index := math.MaxFloat64, 0
			points := make(plotter.XYs, 0, len(particles))
			for i := 0; i < Samples-window; i++ {
				mean, count := 0.0, 0.0
				for j := i; j < i+window; j++ {
					mean += samples[j].Cost
					count++
				}
				mean /= count
				stddev := 0.0
				for j := i; j < i+window; j++ {
					diff := mean - samples[j].Cost
					stddev += diff * diff
				}
				stddev = math.Sqrt(stddev / count)
				if stddev < min {
					min, index = stddev, i+window/2+1
				}
				points = append(points, plotter.XY{X: float64(i), Y: 1 / stddev})
			}
			best = samples[index].Sample
			if s == 0 || s == epochs-1 {
				p := plot.New()
				p.Title.Text = "verse"
				p.X.Label.Text = "x"
				p.Y.Label.Text = "y"

				scatter, err := plotter.NewScatter(points)
				if err != nil {
					panic(err)
				}
				scatter.GlyphStyle.Radius = vg.Length(3)
				scatter.GlyphStyle.Shape = draw.CircleGlyph{}
				p.Add(scatter)

				err = p.Save(8*vg.Inch, 8*vg.Inch, fmt.Sprintf("spectrum_%d.png", s))
				if err != nil {
					panic(err)
				}
			}
		} else {
			for _, sample := range samples {
				if optimizer.Optimize(current, sample.Cost) {
					best = sample.Sample
				}
			}
		}

		index = 0
		for i := length - n; i < length; i++ {
			particles[i] = best[index]
			index++
		}
		for i := length - n; i < length; i++ {
			particle := particles[i]
			particle.T++
			particles = append(particles, particle)
		}
	}

	verse := NewMatrix(0, 4, len(particles))
	for _, particle := range particles {
		verse.Data = append(verse.Data, particle.X, particle.Y, particle.Z, particle.T)
	}
	verse, projection := PCA(verse)
	r, c := projection.Dims()
	fmt.Println(r, c, projection.RawMatrix().Data)
	points := make(plotter.XYs, 0, len(particles))
	for i := 0; i < len(particles); i++ {
		points = append(points, plotter.XY{X: verse.Data[i*4], Y: verse.Data[i*4+1]})
	}
	p := plot.New()
	p.Title.Text = "verse"
	p.X.Label.Text = "x"
	p.Y.Label.Text = "y"

	scatter, err := plotter.NewScatter(points)
	if err != nil {
		panic(err)
	}
	scatter.GlyphStyle.Radius = vg.Length(3)
	scatter.GlyphStyle.Shape = draw.CircleGlyph{}
	p.Add(scatter)
	err = p.Save(16*vg.Inch, 16*vg.Inch, "verse.png")
	if err != nil {
		panic(err)
	}

	if !*Flag3D {
		return
	}

	for i := range particles {
		particles[i].I = i
	}
	nearest := make([]map[int]int, len(particles))
	sum := func() {
		for i := range particles[:len(particles)-1] {
			n := nearest[particles[i].I]
			if n == nil {
				n = make(map[int]int)
				nearest[particles[i].I] = n
			}
			n[particles[i+1].I]++
		}
	}
	sort.Slice(particles, func(i, j int) bool {
		return particles[i].T < particles[j].T
	})
	sum()
	sort.Slice(particles, func(i, j int) bool {
		return particles[i].X < particles[j].X
	})
	sum()
	sort.Slice(particles, func(i, j int) bool {
		return particles[i].Y < particles[j].Y
	})
	sum()
	sort.Slice(particles, func(i, j int) bool {
		return particles[i].Z < particles[j].Z
	})
	sum()
	sort.Slice(particles, func(i, j int) bool {
		return particles[i].T < particles[j].T
	})

	register := make([]Particle, 4)
	for i := 0; i < 4; i++ {
		register[i] = particles[i]
	}
	images := make([]*image.Paletted, epochs)
	index := 0
	for s := 0; s < epochs; s++ {
		for j := 0; j < 4; j++ {
			particle := register[j]
			max, i := 0, 0
			for key, value := range nearest[particle.I] {
				if value > max {
					max = value
					i = key
				}
			}
			_ = i
			register[particles[index].I%4] = particles[index]
			index++
		}
		mesh := gl.NewEmptyMesh()
		for _, particle := range register {
			var x, y, z float64
			x = particle.X
			y = particle.Y
			z = particle.Z
			p := gl.Vector{X: x, Y: y, Z: z}.MulScalar(4)
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
		c := context.Image()
		c = resize.Resize(width, height, c, resize.Bilinear)

		opts := gif.Options{
			NumColors: 256,
			Drawer:    drw.FloydSteinberg,
		}
		bounds := c.Bounds()

		paletted := image.NewPaletted(bounds, palette.Plan9[:opts.NumColors])
		if opts.Quantizer != nil {
			paletted.Palette = opts.Quantizer.Quantize(make(color.Palette, 0, opts.NumColors), c)
		}
		opts.Drawer.Draw(paletted, bounds, c, image.Point{})

		images[s] = paletted
	}

	animation := &gif.GIF{}
	for _, paletted := range images {
		animation.Image = append(animation.Image, paletted)
		animation.Delay = append(animation.Delay, 0)
	}

	filename := "quad.gif"
	f, _ := os.OpenFile(filename, os.O_WRONLY|os.O_CREATE, 0600)
	defer f.Close()
	gif.EncodeAll(f, animation)
}

func main() {
	rng := rand.New(rand.NewSource(1))
	flag.Parse()

	if *FlagQuaternion {
		QuaternionMode(rng)
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
		} else if *FlagConstant {
			optimizer = &ConstantOptimizer{Min: math.MaxFloat64}
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
			current += value
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
				sum += value
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
	} else if *FlagConstant {
		filename = "const.gif"
	}
	f, _ := os.OpenFile(filename, os.O_WRONLY|os.O_CREATE, 0600)
	defer f.Close()
	gif.EncodeAll(f, animation)
}
