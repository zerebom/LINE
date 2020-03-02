# (1) パッケージをインストールする（初回のみ）
install.packages("broom")

# (2) ライブラリの読み出し
library("tidyverse")
library("broom")

# (3) データの読み込み
email_data <- read_csv("http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv")

# (4) 女性向けメールが配信されたデータを削除したデータを作成
male_df <- email_data %>%
  filter(segment != "Womens E-Mail") %>% # 女性向けメールが配信されたデータを削除
  mutate(treatment = ifelse(segment == "Mens E-Mail", 1, 0)) # 介入を表すtreatment変数を追加

# (5) セレクションバイアスのあるデータを作成
## seedを固定
set.seed(1)

## 条件に反応するサンプルの量を半分にする
obs_rate_c <- 0.5
obs_rate_t <- 0.5

## バイアスのあるデータを作成
biased_data <- male_df %>%
  mutate(obs_rate_c =
           ifelse( (history > 300) | (recency < 6) |
                     (channel == "Multichannel"), obs_rate_c, 1),
         obs_rate_t =
           ifelse( (history > 300) | (recency < 6) |
                     (channel == "Multichannel"), 1, obs_rate_t),
         random_number = runif(n = NROW(male_df))) %>%
  filter( (treatment == 0 & random_number < obs_rate_c ) |
            (treatment == 1 & random_number < obs_rate_t) )


#=========================ここまで前回と同じ========================================

# (6) バイアスのあるデータでの回帰分析
## 回帰分析の実行
# spend...購入額、treatment...介入変数、history...過去の購入額
biased_reg <- lm(data = biased_data, formula = spend ~ treatment + history)

## 分析結果のレポート
summary(biased_reg)
# Pr(>|t|)...p値
# β3=treatmentの推定結果は0.9、これが施策の効果。帰無仮説を棄却できる


# tidyはlmで出力される値をdataframeにうまく吐き出す関数
## 推定されたパラメーターの取り出し
# summaryのcoefficientsだけ取り出す。
biased_reg_coef <- tidy(biased_reg)


# (7) RCTデータでの回帰分析とバイアスのあるデータでの回帰分析の比較
# 共変量を加えることで、セレクションバイアスが小さくなっていることを確認する

## RCTデータでの単回帰
rct_reg <- lm(data = male_df, formula = spend ~ treatment)
rct_reg_coef <- tidy(rct_reg)

## バイアスのあるデータでの単回帰
nonrct_reg <- lm(data = biased_data, formula = spend ~ treatment)
nonrct_reg_coef <- tidy(nonrct_reg)

## バイアスのあるデータでの重回帰
nonrct_mreg <- lm(data = biased_data,
                  formula = spend ~ treatment + recency + channel + history)
nonrct_mreg_coef <- tidy(nonrct_mreg)

## 8 OVBの確認
# a) history抜きの回帰分析とパラメータの取り出し
short_coef <- biased_data %>%
  lm(data = .,
     formula = spend ~ treatment + recency + channel) %>%
  tidy()

# aの結果から介入効果に関するパラメータの見取り出す
alpha_1 <- short_coef %>%
  filter(term == "treatment") %>%
  pull(estimate)


## (b) historyを追加した回帰分析とパラメーターの取り出し
long_coef <- biased_data %>%
  lm(data = .,
     formula = spend ~ treatment + recency + channel + history) %>%
  tidy()

## bの結果から介入とhistoryに関するパラメーターを取り出す
beta_1 <- long_coef %>% filter(term == "treatment") %>% pull(estimate)
beta_2 <- long_coef %>% filter(term == "history") %>% pull(estimate)

## (c) 脱落した変数と介入変数での回帰分析
omitted_coef <- biased_data %>%
  lm(data = ., formula = history ~ treatment + channel + recency) %>%
  tidy()
## cの結果から介入変数に関するパラメーターを取り出す
gamma_1 <- omitted_coef %>% filter(term == "treatment") %>% pull(estimate)

## OVBの確認
beta_2*gamma_1
alpha_1 - beta_1


