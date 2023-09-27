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
	"math/rand"
	"os"

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

func main() {
	rng := rand.New(rand.NewSource(1))
	flag.Parse()

	n := 4
	const sets = 8

	particles := []Particle{
		{X: 0, Y: 0},
		{X: 128, Y: 0},
		{X: 0, Y: 128},
		{X: 64, Y: 64},
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
			particle.T += float64(i)
			particles = append(particles, particle)
		}
	}

	getEntropy := func() []float64 {
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
		entropy := getEntropy()
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
			entropy := getEntropy()
			index, sum := 0, 0.0
			for _, value := range entropy {
				sum += -value
			}
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
