# Historical pipeline archive

These files are unchanged copies of the original `naacl/` files at commit
`7e43efc6bd2cd836a1bac71dc4f7aac9b616a8fb` on `naacl-validity-repair`.
The original tests moved to `../tests/` and point here for reference regression testing.

The active pipeline is documented in `../README.md`. Historical Slurm scripts and
runbooks contain their original repository-relative paths and should be run from
an original-branch checkout, not from this subdirectory. Individual Python reference
executors can run here with explicit input/output paths, as the differential tests do.
Do not update these files to implement optimized behavior. The active code imports
its own current modules and does not patch module globals in this archive.
