    library('tidyverse')
    
    email_data<-read_csv('http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv')
    
    # 1.4.1 RCTデータの有意差検定
    
    # %>%で次の関数に値を渡している
    male_df <- email_data %>%
    filter(segment != "Womens E-Mail") %>%
    # mutateで新しい列を作成。
    mutate(treatment = if_else(segment == "Mens E-Mail", 1, 0))
    
    # 集計による比較
    summary_by_segment <- male_df %>%
      # groupbyとsummariseがセット。統計量を作成
      group_by(treatment) %>% 
      #　group毎のconversionの平均
      summarise(conversion_rate = mean(conversion),
                # group毎のspendの平均
                spend_mean = mean(spend),
                count = n())
    
    # 有意差検定
    
    # (a)男性向けメールが配信されたグループの購買データを得る
    mens_mail <- male_df %>%
      filter(treatment == 1) %>%
      # pull...指定した変数をベクトルで受け取る
      pull(spend)
    
    # (b)メールが配信されなかったグループの(略)
    no_mail <- male_df %>%
      filter(treatment == 0) %>%
      pull(spend)
    
    #　介入の有無で分散が異ならないか = TRUE
    rct_ttest <- t.test(mens_mail, no_mail, var.equal = TRUE)
    
    
    # 1.4.3  バイアスのあるデータによる効果の検証
    set.seed(1)
    
    obs_rate_c <- 0.5
    obs_rate_r <- 0.5
    
    ## biasedデータを作成
    # アクティブなユーザにメールを送信した傾向があるデータにする
    biased_data <- male_df %>%
      # obs_rate_c,rを列に追加する。
      mutate(
        # 昨年購入額・最後の購入・接触チャネル複数のユーザの半分を削る(そのための列を用意)
        # if_else, ifなら第二引数、elseなら第三引数の値になる
        obs_rate_c = if_else(
        (history > 300) | (recency < 6) | (channel == 'Multichannel'),
          obs_rate_c , 1),
        # 逆の比率も加える
        obs_rate_r = if_else(
        (history > 300) | (recency < 6) | (channel == 'Multichannel'),
          1, obs_rate_c),
      random_number = runif(n = NROW(male_df))) %>%
    filter((treatment == 0 & random_number < obs_rate_c)|
             (treatment == 1 & random_number < obs_rate_r))
    
    # 同様にt検定を行う
    summary_by_segment_biased <- biased_data %>%
      group_by(treatment) %>%
      summarise(conversion_rate = mean(conversion),
                spend_mean =mean(spend),
                count = n())
    
    mens_mail_biased <- biased_data %>%
      filter(treatment ==1) %>%
      pull(spend)
    
    no_mail_biased <- biased_data %>%
      filter(treatment == 0) %>%
      pull(spend)
    
    rct_ttest_biased <- t.test(mens_mail_biased, no_mail_biased, var.equal = T)
    
    
    
    
    
    
    
    
    
    
 