#import "@preview/codelst:2.0.2": sourcecode, code-frame

#let code-block = block.with(
  stroke: 1pt,
  inset: 0.65em,
  radius: 4pt,
)

#sourcecode[```py
#show "ArtosFlow": name => box[
  #box(image(
    "logo.svg",
    height: 0.7em,
  ))
  #name
]

This report is embedded in the
ArtosFlow project. ArtosFlow is a
project of the Artos Institute.
```]

#sourcecode(
  numbering: "I",
  numbers-style: lno => align(right, [#text(eastern, emph(lno)) |]),
  gutter: 1em,
  tab-size: 8,
  gobble: 1,
  showlines: true,
)[
  ```rust


  	// Function that returns a boolean value
  	fn is_divisible_by(lhs: u32, rhs: u32) -> bool {
  			// Corner case, early return
  			if rhs == 0 {
  					return false;
  			}

  			// This is an expression, the `return` keyword is not necessary here
  			lhs % rhs == 0
  	}


  ```
]

#block(width: 100%)[
  #sourcecode(
    numbers-width: -6mm,
    frame: block.with(width: 75%, fill: rgb("#299281"), inset: 5mm),
  )[```rust
      // Functions that "don't" return a value, actually return the unit type `()`
      fn fizzbuzz(n: u32) -> () {
          if is_divisible_by(n, 15) {
              println!("fizzbuzz");
          } else if is_divisible_by(n, 3) {
              println!("fizz");
          } else if is_divisible_by(n, 5) {
              println!("buzz");
          } else {
              println!("{}", n);
          }
      }
    ```]
  #place(
    top + right,
    block(width: 23%)[
      #set par(justify: true)
      #lorem(40)
    ],
  )
]

#sourcecode(
  numbering: "(1)",
  numbers-side: right,
  numbers-style: lno => text(1.5em, rgb(143, 254, 9), [#sym.arrow.l #lno]),

  frame: code => {
    set text(luma(245))
    code-frame(
      fill: luma(24),
      stroke: 4pt + rgb(143, 254, 9),
      radius: 0pt,
      inset: .65em,
      code,
    )
  },
)[```rust
  // When a function returns `()`, the return type can be omitted from the
  // signature
  fn fizzbuzz_to(n: u32) {
      for n in 1..=n {
          fizzbuzz(n);
      }
  }
  ```]