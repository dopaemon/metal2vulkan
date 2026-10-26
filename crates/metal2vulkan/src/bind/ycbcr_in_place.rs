//! A two-plane YCbCr texture's planes read in place: a function reads the codes from the planes'
//! own bytes and converts each texel as Metal does, where the texture's image of converted texels
//! (`ycbcr_convert`) would otherwise stand between them.
//!
//! **What it covers.** `YCBCR8_420_2P` and `YCBCR10_420_2P` (500 and 505) over a surface's two
//! planes, read converted ([`YCbCrOutput::Converted`]): the 4:2:0 layouts of unpacked codes, a
//! byte or the high 10 bits of a little-endian 16-bit word. A texture whose planes are those
//! ([`YCbCrInPlace::of`]) states it in its facts (`TextureFacts::ycbcr_in_place`), and a texture
//! slot whose function reads it only as the in-place form reads
//! (`RuntimeFact::YCbCrInPlace`) takes the slot's in-place variant (`VariantKey::ycbcr_in_place`).
//!
//! **The binding.** The slot's one descriptor is a uniform texel buffer over the planes' bytes,
//! from the first byte of the first plane in memory, in [`YCbCrInPlace::format`]: `R8G8_UINT`
//! for 8-bit codes and `R16G16_UINT` for 10-bit ones, so one element holds two luma codes or one
//! chroma pair. Vulkan requires `VK_FORMAT_FEATURE_UNIFORM_TEXEL_BUFFER_BIT` of both formats on
//! every device (the specification's "Required Format Support": `R8G8_UINT` in "1-3 Byte-Sized
//! Components", `R16G16_UINT` in "16-bit Components"). The texture's size, its planes' places
//! and rows in elements, and its conversion are the slot's push field
//! (`PushField::YCbCrInPlace`, [`YCbCrInPlace::words`]).
//!
//! **What a function reads** is Metal's texel (`ycbcr_convert`'s module docs): the luma code at
//! the texel's place, the four nearest chroma pairs weighted 9/3/3/1 and rounded to a code, and
//! the conversion of the surface's range and matrix, each channel a step of `1 / 2048`, alpha 1.
//! A sample is Metal's filter of those texels on that grid (`lower::sampler::in_place`), exact,
//! as the image's emulated filter is.

use super::ycbcr_convert::{YCbCrMatrix, YCbCrOutput, YCbCrPlanes, YCbCrRange};
use crate::metal::format::ChromaSubsampling;
use crate::vk;

/// A two-plane YCbCr texture's planes as a function reads them in place, where the texture's image
/// of converted texels would otherwise stand between them: the words of its slot's push field
/// (`PushField::YCbCrInPlace`), and the uniform texel buffer the slot binds over the planes' bytes
/// ([`format`](Self::format), from the first byte of the first plane for
/// [`view_bytes`](Self::view_bytes)). A texture states one in its facts
/// (`TextureFacts::ycbcr_in_place`), and a slot whose function reads the texture only as this form
/// does (`RuntimeFact::YCbCrInPlace`) takes the in-place variant (`VariantKey::ycbcr_in_place`),
/// which fetches the codes and converts and filters them as Metal does: the luma code at the
/// texel's place, the four nearest chroma pairs weighted 9/3/3/1 and rounded to a code, the
/// surface's range and matrix, each channel a step of `1 / 2048`, alpha 1. Vulkan requires
/// `VK_FORMAT_FEATURE_UNIFORM_TEXEL_BUFFER_BIT` of both formats on every device.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct YCbCrInPlace {
    words: [u32; YCbCrInPlace::WORDS],
    /// The bytes the texel buffer views, from the planes' first byte: the end of the last row of
    /// codes either plane holds, rounded up to a whole element.
    bytes: u64,
}

impl YCbCrInPlace {
    /// The words of the slot's push field ([`words`](Self::words)), two `uvec4`s.
    pub const WORDS: usize = 8;
    /// Word: the texture's width in texels.
    pub const WIDTH: usize = 0;
    /// Word: the texture's height in texels.
    pub const HEIGHT: usize = 1;
    /// Word: the luma plane's first element in the texel buffer.
    pub const LUMA_AT: usize = 2;
    /// Word: the elements from one luma row to the next.
    pub const LUMA_ROW: usize = 3;
    /// Word: the chroma plane's first element in the texel buffer.
    pub const CHROMA_AT: usize = 4;
    /// Word: the elements from one chroma row to the next.
    pub const CHROMA_ROW: usize = 5;
    /// Word: the conversion: the codes' bits in bits 0 to 7, full range in bit 8, and the matrix
    /// in bits 9 and 10 (0 BT.601, 1 BT.709, 2 BT.2020).
    pub const CONVERSION: usize = 6;

    /// The in-place form of `planes`, where a function reads them so: a
    /// converted 4:2:0 layout of unpacked 8- or 10-bit codes, both planes in the bytes (a chroma
    /// plane of rows; not the conversion's row of zeros), each plane's first byte and rows a
    /// multiple of an element. `None` for any other: such a texture keeps its image of converted
    /// texels alone.
    pub fn of(planes: &YCbCrPlanes) -> Option<YCbCrInPlace> {
        let codes = planes.codes;
        if codes.output != YCbCrOutput::Converted
            || codes.subsampling != ChromaSubsampling::Chroma420
            || codes.packed
            || !matches!((codes.bits, codes.bytes), (8, 1) | (10, 2))
            || planes.chroma[1] == 0
        {
            return None;
        }
        let element = 2 * codes.bytes;
        let [luma_at, luma_row] = planes.luma;
        let [chroma_at, chroma_row] = planes.chroma;
        if [luma_at, luma_row, chroma_at, chroma_row]
            .iter()
            .any(|v| v % element != 0)
        {
            return None;
        }
        let [w, h] = planes.size;
        if w == 0 || h == 0 {
            return None;
        }
        let [cw, ch] = planes.chroma_size().map(u64::from);
        let unit = u64::from(codes.bytes);
        let end = |at: u32, row: u32, rows: u64, bytes: u64| {
            u64::from(at) + u64::from(row) * (rows - 1) + bytes
        };
        let reach = end(luma_at, luma_row, u64::from(h), unit * u64::from(w)).max(end(
            chroma_at,
            chroma_row,
            ch,
            unit * 2 * cw,
        ));
        let bytes = reach.next_multiple_of(u64::from(element));
        let mut words = [0; Self::WORDS];
        words[Self::WIDTH] = w;
        words[Self::HEIGHT] = h;
        words[Self::LUMA_AT] = luma_at / element;
        words[Self::LUMA_ROW] = luma_row / element;
        words[Self::CHROMA_AT] = chroma_at / element;
        words[Self::CHROMA_ROW] = chroma_row / element;
        words[Self::CONVERSION] = conversion_word(codes.bits, planes.range, planes.matrix);
        Some(YCbCrInPlace { words, bytes })
    }

    /// The words of the slot's push field.
    pub fn words(&self) -> [u32; Self::WORDS] {
        self.words
    }

    /// The codes' bits: 8 or 10.
    pub fn bits(&self) -> u32 {
        self.words[Self::CONVERSION] & 0xff
    }

    /// The texel buffer's format: `R8G8_UINT` of 8-bit codes, `R16G16_UINT` of 10-bit ones.
    pub fn format(&self) -> vk::Format {
        match self.bits() {
            8 => vk::Format::R8G8_UINT,
            _ => vk::Format::R16G16_UINT,
        }
    }

    /// The bytes of one element of the texel buffer: 2 of 8-bit codes, 4 of 10-bit ones.
    pub fn element_bytes(&self) -> u64 {
        match self.bits() {
            8 => 2,
            _ => 4,
        }
    }

    /// The bytes the texel buffer views from the planes' first byte: through the last row of
    /// codes either plane holds, a whole number of elements.
    pub fn view_bytes(&self) -> u64 {
        self.bytes
    }

    /// The elements the texel buffer holds (`maxTexelBufferElements` bounds them).
    pub fn elements(&self) -> u64 {
        self.bytes / self.element_bytes()
    }
}

/// The conversion word ([`YCbCrInPlace::CONVERSION`]) of `bits`-bit codes of `range` and
/// `matrix`.
fn conversion_word(bits: u32, range: YCbCrRange, matrix: YCbCrMatrix) -> u32 {
    let full = match range {
        YCbCrRange::Video => 0,
        YCbCrRange::Full => 1,
    };
    let m = match matrix {
        YCbCrMatrix::Bt601 => 0,
        YCbCrMatrix::Bt709 => 1,
        YCbCrMatrix::Bt2020 => 2,
    };
    bits | full << 8 | m << 9
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bind::ycbcr_convert::YCbCrCodes;
    use crate::metal::format::ycbcr_format;
    use crate::metal::PixelFormat;

    fn planes(pf: PixelFormat, luma: [u32; 2], chroma: [u32; 2]) -> YCbCrPlanes {
        let codes = ycbcr_format(pf).and_then(YCbCrCodes::of).expect("codes");
        YCbCrPlanes::new(
            [7, 5],
            luma,
            chroma,
            (YCbCrRange::Full, YCbCrMatrix::Bt709),
            codes,
        )
    }

    /// The two converted 4:2:0 formats take the form, in elements of their own width; every
    /// other layout keeps its image.
    #[test]
    fn only_converted_four_two_zero_codes_read_in_place() {
        let p8 =
            YCbCrInPlace::of(&planes(PixelFormat::YCBCR84202P, [0, 16], [96, 16])).expect("8-bit");
        assert_eq!(p8.format(), vk::Format::R8G8_UINT);
        assert_eq!(
            p8.words(),
            [
                7,
                5,
                0,
                8,
                48,
                8,
                conversion_word(8, YCbCrRange::Full, YCbCrMatrix::Bt709),
                0
            ]
        );
        // The chroma plane's last row (3 rows of 4 pairs) ends at 96 + 32 + 8.
        assert_eq!(p8.view_bytes(), 136);
        assert_eq!(p8.elements(), 68);
        let p10 = YCbCrInPlace::of(&planes(PixelFormat::YCBCR104202P, [0, 16], [96, 32]))
            .expect("10-bit");
        assert_eq!(p10.format(), vk::Format::R16G16_UINT);
        assert_eq!(p10.words()[YCbCrInPlace::LUMA_ROW], 4);
        assert_eq!(p10.words()[YCbCrInPlace::CHROMA_AT], 24);
        assert_eq!(p10.bits(), 10);
        // The luma plane ends the bytes: its last row's seven codes round up to a whole element.
        let tail = YCbCrInPlace::of(&planes(PixelFormat::YCBCR84202P, [64, 16], [0, 16]))
            .expect("chroma first");
        assert_eq!(tail.view_bytes(), 64 + 4 * 16 + 8);
        for pf in [
            PixelFormat::YCBCR84222P,
            PixelFormat::YCBCR84442P,
            PixelFormat::YCBCR84202PSRGB,
            PixelFormat::RGB84202P,
            PixelFormat::YCBCR104202PPACKED,
        ] {
            assert_eq!(
                YCbCrInPlace::of(&planes(pf, [0, 16], [96, 16])),
                None,
                "{pf:?}"
            );
        }
        // The conversion's row of zeros is no chroma plane.
        let zero = planes(PixelFormat::YCBCR84202P, [0, 16], [96, 0]);
        assert_eq!(YCbCrInPlace::of(&zero), None);
    }
}
