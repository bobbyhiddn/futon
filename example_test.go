package futon_test

import (
	"context"
	"fmt"
	"log"

	"github.com/bobbyhiddn/futon"
)

// Embed two texts and rank them against a query. Requires a pack on disk:
// see the README for how to build one.
func Example() {
	e, err := futon.New(futon.Options{Model: "bge-small-en-v1.5"})
	if err != nil {
		log.Fatal(err) // *packs.NotInstalledError if the pack is missing
	}
	defer e.Close()

	ctx := context.Background()
	docs, err := e.EmbedBatch(ctx, []string{
		"Go is a statically typed, compiled language.",
		"Sourdough needs a long, slow fermentation.",
	})
	if err != nil {
		log.Fatal(err)
	}
	query, err := e.Embed(ctx, "which language compiles to machine code?")
	if err != nil {
		log.Fatal(err)
	}
	for i, d := range docs {
		fmt.Printf("doc %d: %.2f\n", i, futon.CosineSimilarity(query, d))
	}
}
