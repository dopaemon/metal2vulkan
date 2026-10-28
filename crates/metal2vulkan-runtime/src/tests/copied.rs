//! The comparisons that keep a copied host buffer coherent with the caller's pages (`copied.rs`),
//! on plain memory: a commit's copy in writes the bytes the CPU changed alone, and a completion's
//! merge back writes the bytes the GPU changed alone, keeping a byte the CPU changed since.

use crate::copied::sync;
use std::ptr::NonNull;

/// `bytes` as a pointer the comparison reads or writes.
fn at(bytes: &mut [u8]) -> NonNull<u8> {
    NonNull::new(bytes.as_mut_ptr()).unwrap()
}

#[test]
fn copied_copy_in_writes_the_bytes_the_cpu_changed_alone() {
    let mut snapshot: Vec<u8> = (0..37).collect();
    let mut pages = snapshot.clone();
    // The GPU changed byte 1 of the copy, not yet merged back.
    let mut copy = snapshot.clone();
    copy[1] = 0xEE;
    // The CPU changed byte 2, in the same word, and byte 36, in the last, short span.
    pages[2] = 0xAA;
    pages[36] = 0xBB;
    // SAFETY: three buffers of the snapshot's length.
    let written = unsafe { sync(at(&mut pages), at(&mut copy), &mut snapshot, false) };
    assert_eq!(written, 2);
    assert_eq!(copy[1], 0xEE, "the GPU's byte in the same word stays");
    assert_eq!((copy[2], copy[36]), (0xAA, 0xBB), "the CPU's bytes are in");
    assert_eq!((snapshot[2], snapshot[36]), (0xAA, 0xBB));
    assert_eq!(snapshot[1], 1, "the GPU's byte is not agreed yet");
}

#[test]
fn copied_merge_back_keeps_a_byte_the_cpu_changed_since() {
    let mut snapshot: Vec<u8> = (0..16).collect();
    let mut pages = snapshot.clone();
    let mut copy = snapshot.clone();
    // The GPU wrote bytes 3, 4 and 9; the CPU wrote byte 4 too, and byte 5, since the commit.
    copy[3] = 0x33;
    copy[4] = 0x44;
    copy[9] = 0x99;
    pages[4] = 0xC4;
    pages[5] = 0xC5;
    // SAFETY: three buffers of the snapshot's length.
    let written = unsafe { sync(at(&mut copy), at(&mut pages), &mut snapshot, true) };
    assert_eq!(written, 2, "bytes 3 and 9");
    assert_eq!(
        (pages[3], pages[9]),
        (0x33, 0x99),
        "the GPU's bytes are back"
    );
    assert_eq!((pages[4], pages[5]), (0xC4, 0xC5), "the CPU's bytes stay");
    // The next copy in carries the CPU's byte 4 (and 5) into the copy.
    // SAFETY: as above.
    let written = unsafe { sync(at(&mut pages), at(&mut copy), &mut snapshot, false) };
    assert_eq!(written, 2);
    assert_eq!((copy[4], copy[5]), (0xC4, 0xC5));
    assert_eq!(snapshot, pages, "they agree");
    assert_eq!(copy, pages);
}
