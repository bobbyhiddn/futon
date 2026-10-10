// Command futon embeds text with a local weight pack and manages packs.
//
//	futon embed [-model id] [-models-dir dir] [-dims n] [text ...]
//	futon packs available [-source url]
//	futon packs pull [-models-dir dir] [-source url] <id> ...
//	futon packs list [-models-dir dir]
//	futon packs verify [-models-dir dir] <id>
//	futon packs install [-models-dir dir] [-force] <archive.futonw>
//	futon packs load [-models-dir dir] [-force] <pack-dir>
//
// embed prints one JSON array per input text. With no text arguments it reads
// one text per line from standard input.
package main

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"os/signal"

	"github.com/bobbyhiddn/futon"
	"github.com/bobbyhiddn/futon/hub"
	"github.com/bobbyhiddn/futon/packs"
)

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}
	var err error
	switch os.Args[1] {
	case "embed":
		err = cmdEmbed(os.Args[2:])
	case "packs":
		err = cmdPacks(os.Args[2:])
	case "-h", "--help", "help":
		usage()
		return
	default:
		usage()
		os.Exit(2)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "futon:", err)
		os.Exit(1)
	}
}

func usage() {
	fmt.Fprint(os.Stderr, `usage:
  futon embed [-model id] [-models-dir dir] [-dims n] [text ...]
  futon packs available [-source url]
  futon packs pull [-models-dir dir] [-source url] <id> ...
  futon packs list [-models-dir dir]
  futon packs verify [-models-dir dir] <id>
  futon packs install [-models-dir dir] [-force] <archive.futonw>
  futon packs load [-models-dir dir] [-force] <pack-dir>

Packs are built with scripts/build-weight-pack.py (see docs/packs.md).
`)
}

func cmdEmbed(args []string) error {
	fs := flag.NewFlagSet("embed", flag.ExitOnError)
	model := fs.String("model", futon.DefaultModel, "pack id")
	modelsDir := fs.String("models-dir", "", "directory holding packs (default "+futon.DefaultModelsDir()+")")
	dims := fs.Int("dims", 0, "truncate to this many dimensions (matryoshka models only)")
	fs.Parse(args)

	e, err := futon.New(futon.Options{Model: *model, ModelsDir: *modelsDir, Dimensions: *dims})
	if err != nil {
		return err
	}
	defer e.Close()

	var texts []string
	if fs.NArg() > 0 {
		texts = fs.Args()
	} else {
		sc := bufio.NewScanner(os.Stdin)
		sc.Buffer(make([]byte, 1<<20), 1<<24)
		for sc.Scan() {
			texts = append(texts, sc.Text())
		}
		if err := sc.Err(); err != nil {
			return err
		}
	}
	if len(texts) == 0 {
		return errors.New("no text to embed")
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	vecs, err := e.EmbedBatch(ctx, texts)
	if err != nil {
		return err
	}
	enc := json.NewEncoder(os.Stdout)
	for _, v := range vecs {
		if err := enc.Encode(v); err != nil {
			return err
		}
	}
	return nil
}

func cmdPacks(args []string) error {
	if len(args) == 0 {
		usage()
		return errors.New("packs: missing subcommand")
	}
	fs := flag.NewFlagSet("packs "+args[0], flag.ExitOnError)
	modelsDir := fs.String("models-dir", "", "directory holding packs (default "+futon.DefaultModelsDir()+")")
	force := fs.Bool("force", false, "replace an already installed pack")
	source := fs.String("source", "", "directory URL to download packs from (default: the hosted models, see MODELS.md)")
	fs.Parse(args[1:])
	dir := *modelsDir
	if dir == "" {
		dir = futon.DefaultModelsDir()
	}

	switch args[0] {
	case "available":
		ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
		defer stop()
		models, err := hub.Available(ctx, hub.Options{BaseURL: *source})
		if err != nil {
			return err
		}
		for _, m := range models {
			fmt.Println(m.ID)
		}
	case "pull":
		if fs.NArg() == 0 {
			return errors.New("packs pull: name at least one pack (see: futon packs available)")
		}
		ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
		defer stop()
		for _, id := range fs.Args() {
			if err := pullOne(ctx, id, dir, *source); err != nil {
				return err
			}
		}
	case "list":
		infos, err := packs.List(dir)
		if err != nil {
			return err
		}
		if len(infos) == 0 {
			fmt.Printf("no packs in %s\n", dir)
			return nil
		}
		for _, in := range infos {
			status := "ok"
			if !in.OK {
				status = fmt.Sprintf("FAILED: %v", in.Err)
			}
			fmt.Printf("%-34s %8.1f MB  %-12s %s\n", in.PackID, float64(in.SizeBytes)/1e6, in.License, status)
		}
	case "verify":
		if fs.NArg() != 1 {
			return errors.New("packs verify: want exactly one pack id")
		}
		in, err := packs.Verify(dir, fs.Arg(0))
		if err != nil {
			return err
		}
		fmt.Printf("%s: verified (%s)\n", in.PackID, in.Dir)
	case "install":
		if fs.NArg() != 1 {
			return errors.New("packs install: want exactly one archive path")
		}
		out, err := packs.Install(dir, fs.Arg(0), *force)
		if err != nil {
			return err
		}
		fmt.Println("installed", out)
	case "load":
		if fs.NArg() != 1 {
			return errors.New("packs load: want exactly one pack directory")
		}
		out, err := packs.LoadDir(dir, fs.Arg(0), *force)
		if err != nil {
			return err
		}
		fmt.Println("installed", out)
	default:
		usage()
		return fmt.Errorf("packs: unknown subcommand %q", args[0])
	}
	return nil
}

// pullOne downloads and installs one pack, reporting progress on stderr.
func pullOne(ctx context.Context, id, modelsDir, source string) error {
	lastPct := -1
	_, err := hub.Pull(ctx, id, hub.Options{
		ModelsDir: modelsDir,
		BaseURL:   source,
		Progress: func(done, total int64) {
			if total <= 0 {
				return
			}
			if pct := int(done * 100 / total); pct != lastPct && pct%5 == 0 {
				lastPct = pct
				fmt.Fprintf(os.Stderr, "\r%s: %3d%% of %.0f MB", id, pct, float64(total)/1e6)
			}
		},
	})
	fmt.Fprintln(os.Stderr)
	var already *packs.AlreadyInstalledError
	if errors.As(err, &already) {
		return fmt.Errorf("%s is already installed with different contents; remove it first to replace it", id)
	}
	if err != nil {
		return err
	}
	fmt.Printf("installed %s in %s\n", id, modelsDir)
	return nil
}
