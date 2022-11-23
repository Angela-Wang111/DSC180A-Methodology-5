def plot_both_loss(all_train_loss, all_val_loss):
    plt.figure(figsize=(20, 10))
    sns.set(style="whitegrid")
    epoch_num = len(all_train_loss)
    
    df = pd.DataFrame({'x':range(epoch_num),
                    'train_loss':all_train_loss,
                      'val_loss':all_val_loss})
    df = df.set_index('x')
    
    train_val_loss = sns.lineplot(data=df, linewidth=2.5)

    ## now label the y- and x-axes.
    plt.ylabel('Customize MAE Loss')
    plt.xlabel('Epoch Number')
    plt.title('MAE Loss')
    plt.show()
    
    fig = train_val_loss.get_figure()
    fig.save('train_val_loss.png')
    
    
def test_model(model, loader):
    n = 0
    y_test = []
    y_true = []
    all_test_loss = []
    model.eval()
    for i, (data, labels) in enumerate(loader):
        if torch.cuda.is_available():
            data, labels = data.cuda(), labels.cuda()

        target = model(data)

        #loss = torch.abs(torch.tensor(1 + labels) - (1 + target)).mean()
        #loss = torch.abs(torch.tensor(labels) - target).mean()
        
        #all_test_loss.append(loss)

        y_test.append(target.detach().cpu())
        y_true.append(labels[0].detach().cpu())
                
    return np.array(y_test), np.array(y_true)#, all_test_loss


def test_mae(y_test, y_true):
    test_mae = np.abs(y_test - y_true).mean()
    
    return test_mae



def plot_pearson_r(y_test, y_true, color = "#4CB391"):
    corr, _ = pearsonr(y_true, y_test)
    
    pearson_plot = sns.scatterplot(x=y_true, y=y_test, color=color)
    plt.title('r = {}'.format(np.around(corr, 3)))
    
    fig = pearson_plot.get_figure()
    fig.savefig("pearson_r.png")