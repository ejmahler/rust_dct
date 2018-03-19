# Release 0.2.0
 - All of the `DCT#ViaFFT` algorithms now allocate a Vec for internal scratch space, rather than using a member variable, allowing them to be immutable.
 - [Breaking Change] The `process` method of all DCT traits take `&self` instead of `&mut self`
 - [Breaking Change] Because all DCT instances are now immutable, all DCT instances that depend on other DCT instances take `Arc` pointers instead of `Box` pointers
 - [Breaking Change] Because all DCT instances are now immutable, the DCT planner now creates and stores `Arc` pointers to DCT instances instead of `Box`, and has a cache of algorithm instances for each DCT type
 - Implemented a "split radix" algorithm for DCT2 and DCT3 that processes power-of-two DCTs much faster than converting them to FFTs
 - Used the split radix alogirthm to derive several hardcoded "butterfly" algorithms for small DCT2 and DCT3 sizes (specifically, sizes 2,4,8,16)
 - [Breaking Change] Deleted the obsolete `DCT4ViaFFT` algorithm. For odd sizes, use `DCT4ViaFFTOdd` instead. For even sizes, use `DCT4ViaDCT3`. Both are considerably faster and use considerably less memory in all scenarios.
 - Lots of style improvements
# Prior releases
No prior release notes were kept.
