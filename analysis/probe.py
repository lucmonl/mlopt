from sklearn.linear_model import LogisticRegression

def transformer_probe(graphs, model, train_loader, test_loader, device, zero_out_attn=False):
    logisticRegr = LogisticRegression()
    
    cls_tokens, targets = model.get_cls_tokens(train_loader, device, zero_out_attn)
    cls_tokens_test, targets_test = model.get_cls_tokens(train_loader, device, zero_out_attn)
    for layer in range(len(cls_tokens)):
        
        x_train, y_train =cls_tokens[layer], targets
        logisticRegr.fit(x_train, y_train)

        x_test, y_test =cls_tokens_test[layer], targets_test
        score = logisticRegr.score(x_test, y_test)
        print(layer, score)
        graphs.layer_cls_score.append(score)