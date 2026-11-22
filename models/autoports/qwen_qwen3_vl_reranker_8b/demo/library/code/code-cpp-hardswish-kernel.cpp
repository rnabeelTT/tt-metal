void kernel_main() {
    uint32_t num_tiles = get_arg(args::num_tiles);

    compute_kernel_hw_startup(dfb::in, dfb::out);

    ckl::eltwise_chain(
        ckl::IterationShape::tiles(num_tiles),
        ckl::CopyTile<
            ckl::input(
                dfb::in,
                ckl::WaitPolicy::PerTile,
                kIsInt ? ckl::PopPolicy::PerTile : ckl::PopPolicy::None,
                ckl::DataFormatReconfig::Disabled),
            ckl::Dst::D0>{},
        ckl::Hardsigmoid<ckl::Dst::D0>{},
        ckl::Optional<
            kIsFloat32,
            ckl::CopyTile<
                ckl::input(dfb::in, ckl::WaitPolicy::None, ckl::PopPolicy::PerTile, ckl::DataFormatReconfig::Disabled),
                ckl::Dst::D1>>{},
        ckl::Optional<kIsFloat32, ckl::MulBinary<ckl::Dst::D0, ckl::Dst::D1, ckl::Dst::D0>>{},
        ckl::Optional<
            kIsFloat,
            ckl::DestReuseBinary<
                ckl::BinaryFpuOp::Mul,
                ckl::input(dfb::in, ckl::WaitPolicy::None, ckl::PopPolicy::PerTile, ckl::DataFormatReconfig::Disabled),
                ckl::DestReuseType::DEST_TO_SRCA>>{},
        ckl::PackTile<ckl::output(
            dfb::out, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, ckl::DataFormatReconfig::Disabled)>{});
}
