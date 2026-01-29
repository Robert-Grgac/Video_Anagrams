from video_anagrams.schedules import AnagramSchedule


def test_schedule_boundaries():
    sched = AnagramSchedule(joint_steps=3)

    assert sched.stage(0) == "joint"
    assert sched.stage(1) == "joint"
    assert sched.stage(2) == "joint"
    assert sched.stage(3) == "anagram"
    assert sched.stage(10) == "anagram"


def test_schedule_helpers():
    sched = AnagramSchedule(joint_steps=2)

    assert sched.is_joint(0)
    assert sched.is_joint(1)
    assert not sched.is_joint(2)

    assert not sched.is_anagram(0)
    assert sched.is_anagram(2)
