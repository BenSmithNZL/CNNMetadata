import config
import torch


def evaluate_baseline(dataset, train_index, test_index, network, class_weights):

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_OF_WORKERS,
        sampler=torch.utils.data.SubsetRandomSampler(train_index))

    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_OF_WORKERS,
        sampler=torch.utils.data.SubsetRandomSampler(test_index))

    model = network
    model = model.to(config.DEVICE)

    class_weights = torch.Tensor(class_weights)
    class_weights = class_weights.to(config.DEVICE)

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.LEARNING_RATE,
        momentum=config.MOMENTUM)

    for epoch in range(config.EPOCHS):
        print(f'Epoch {epoch + 1} of {config.EPOCHS}')

        for data in train_loader:
            inputs, labels = data
            inputs = inputs.to(config.DEVICE)
            labels = labels.to(config.DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    predicted = []
    true = []
    model.eval()

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            inputs = inputs.to(config.DEVICE)
            labels = labels.to(config.DEVICE)

            outputs = model(images)
            predicted.extend(torch.max(outputs, 1)[1].tolist())
            true.extend(labels.tolist())

    return(predicted, true)


def evaluate_metadata(dataset, train_index, test_index, network, class_weights):

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_OF_WORKERS,
        sampler=torch.utils.data.SubsetRandomSampler(train_index))

    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_OF_WORKERS,
        sampler=torch.utils.data.SubsetRandomSampler(test_index))

    model = network
    model = model.to(config.DEVICE)

    class_weights = torch.Tensor(class_weights)
    class_weights = class_weights.to(config.DEVICE)

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.LEARNING_RATE,
        momentum=config.MOMENTUM)

    for epoch in range(config.EPOCHS):
        print(f'Epoch {epoch + 1} of {config.EPOCHS}')

        for data in train_loader:
            inputs, labels, metadata = data
            inputs = inputs.to(config.DEVICE)
            labels = labels.to(config.DEVICE)
            metadata = metadata.to(config.DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs, metadata)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    predicted = []
    true = []
    model.eval()

    with torch.no_grad():
        for data in test_loader:
            images, labels, metadata = data
            inputs = inputs.to(config.DEVICE)
            labels = labels.to(config.DEVICE)
            metadata = metadata.to(config.DEVICE)

            outputs = model(images, metadata)
            predicted.extend(torch.max(outputs, 1)[1].tolist())
            true.extend(labels.tolist())

    return(predicted, true)
