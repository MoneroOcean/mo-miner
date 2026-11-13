/* NVRTC/PTX archives still emit legacy .ctors. Modern startup code consumes only
 * .init_array; collect both as GNU ld does, including their priority ordering.
 * Native LLD links need this explicitly, not manual calls into compiler internals. */
SECTIONS {
  .init_array : {
    KEEP(*(SORT_BY_INIT_PRIORITY(.init_array.*) SORT_BY_INIT_PRIORITY(.ctors.*)))
    KEEP(*(.init_array EXCLUDE_FILE (*crtbegin.o *crtbegin?.o *crtend.o *crtend?.o) .ctors))
  }
}
INSERT BEFORE .fini_array;
