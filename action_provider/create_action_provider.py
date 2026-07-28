def create_action_provider(env,args):
    """create action provider based on parameters"""
    if args.action_source == "dds":
        from action_provider.action_provider_dds import DDSActionProvider

        return DDSActionProvider(
            env=env,
            args_cli=args
        )
    elif args.action_source == "dds_wholebody":
        from action_provider.action_provider_wh_dds import DDSRLActionProvider

        return DDSRLActionProvider(
            env=env,
            args_cli=args
        )
    elif args.action_source == "replay":
        from action_provider.action_provider_replay import FileActionProviderReplay

        return FileActionProviderReplay(env=env,args_cli=args)
    elif args.action_source == "pose_grasp":
        from action_provider.action_provider_pose_grasp import PoseGraspActionProvider

        return PoseGraspActionProvider(
            env=env,
            args_cli=args,
        )
    elif args.action_source == "camera_pose":
        from action_provider.action_provider_camera_pose import CameraPoseActionProvider

        return CameraPoseActionProvider(
            env=env,
            args_cli=args,
        )
    elif args.action_source == "vr_ik":
        from action_provider.action_provider_vr_ik import R1A7VRDualArmIKProvider

        return R1A7VRDualArmIKProvider(
            env=env,
            args_cli=args,
        )
    else:
        print(f"unknown action source: {args.action_source}")
        return None
