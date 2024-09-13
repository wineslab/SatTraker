from telescope import Telescope


def test_telescope():
    telescope = Telescope()
    telescope.calibrate()
    assert telescope.target.name == "Sun"
    
if __name__ == "__main__":
    test_telescope()