PIN_MEMORY = True

MAX_THREADING = 40

gtsrb_mean = [0.3403, 0.3121, 0.3214]
gtsrb_std = [0.2724, 0.2608, 0.2669]

mnist_mean = (0.13066373765468597,)
mnist_std = (0.30810782313346863,)

cifar10_mean = [0.4914672374725342, 0.4822617471218109, 0.4467701315879822]  # [0.4914, 0.4822, 0.4465]
cifar10_std = [0.24703224003314972, 0.24348513782024384, 0.26158785820007324]  # [0.2023, 0.1994, 0.2010]

tiny_imagenet_mean = [0.485, 0.456, 0.406]  # [0.4789886474609375, 0.4457630515098572, 0.3944724500179291]
tiny_imagenet_std = [0.229, 0.224, 0.225]  # [0.27698642015457153, 0.2690644860267639, 0.2820819020271301]


transform_dict = dict(gtsrb=(gtsrb_mean, gtsrb_std),
                      mnist=(mnist_mean, mnist_std),
                      cifar10=(cifar10_mean, cifar10_std),
                      tinyimagenet=(tiny_imagenet_mean, tiny_imagenet_std)
                      )