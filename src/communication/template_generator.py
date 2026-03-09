"""
Template Generator - Creates personalized, bilingual message templates

Dynamically generates templates from KB-extracted features and Octalysis hooks.
Each template row has BOTH languages in separate columns:
  - message_title_en, message_title_hi
  - message_body_en, message_body_hi
  - cta_text_en, cta_text_hi

Hindi columns use Hinglish (default secondary language, configurable).
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import random


class TemplateGenerator:
    """Generates personalized message templates with bilingual content.

    Domain-generic: All templates are generated dynamically from KB-extracted
    features and Octalysis hooks rather than hardcoded strings.
    """

    def __init__(self, knowledge_bank: Dict, themes: pd.DataFrame):
        self.kb = knowledge_bank
        self.themes = themes
        self.templates = None

        # Extract feature names from KB for use in templates
        self.feature_names = self._extract_feature_names()
        self.domain = knowledge_bank.get('detected_domain', 'generic')

    def _extract_feature_names(self) -> List[str]:
        """Extract feature names from the knowledge bank."""
        features = []
        fgm = self.kb.get('feature_goal_map', {})
        for f in fgm.get('features', []):
            features.append(f.get('feature_name', 'Core Feature'))
        if not features:
            features = ['Core Feature']
        return features

    def generate_templates(self, segment_goals: pd.DataFrame) -> pd.DataFrame:
        """
        Generate 5 message templates for each segment × lifecycle × goal × theme

        Each template row contains BOTH English and Hinglish in separate columns.

        Args:
            segment_goals: DataFrame with segment goals

        Returns:
            pd.DataFrame: Message templates with bilingual columns
        """
        print("\n[Edit]  Generating message templates...")

        templates = []
        template_id = 1

        # Group by segment, lifecycle, goal
        for (seg_id, lifecycle, goal), group in segment_goals.groupby(
            ['segment_id', 'lifecycle_stage', 'primary_goal']
        ):
            seg_name = group['segment_name'].iloc[0]

            # Get themes for this segment × lifecycle
            theme_row = self.themes[
                (self.themes['segment_id'] == seg_id) &
                (self.themes['lifecycle_stage'] == lifecycle)
            ]

            if theme_row.empty:
                continue

            primary_theme = theme_row['primary_theme'].iloc[0]
            secondary_theme = None
            if 'secondary_theme' in theme_row.columns:
                secondary_theme = theme_row['secondary_theme'].iloc[0]

            theme_values = [primary_theme]
            if pd.notna(secondary_theme) and secondary_theme != primary_theme:
                theme_values.append(secondary_theme)

            # Generate 5 variants per theme
            for theme in theme_values:
                for variant in range(5):
                    # Generate bilingual content (title, body, CTA)
                    title_en, title_hi = self._generate_title(theme, variant)
                    body_en, body_hi = self._generate_body(seg_name, lifecycle, goal, theme, variant)
                    cta_en, cta_hi = self._generate_cta(goal, theme, variant)

                    # Determine tone
                    tone = self._select_tone(lifecycle, theme)

                    # Determine feature reference
                    feature = self._select_feature(goal, theme)

                    # Single row with both languages
                    templates.append({
                        'template_id': f'TPL_{template_id:04d}',
                        'segment_id': seg_id,
                        'segment_name': seg_name,
                        'lifecycle_stage': lifecycle,
                        'goal': goal,
                        'theme': theme,
                        'variant': variant + 1,
                        'message_title_en': title_en,
                        'message_title_hi': title_hi,
                        'message_body_en': body_en,
                        'message_body_hi': body_hi,
                        'cta_text_en': cta_en,
                        'cta_text_hi': cta_hi,
                        'tone': tone,
                        'hook': theme,
                        'feature_reference': feature
                    })

                    template_id += 1

        self.templates = pd.DataFrame(templates)

        print(f"   [OK] Generated {len(templates)} bilingual templates")
        print(f"   [OK] Each template has English + Hinglish columns")

        return self.templates

    def _generate_title(self, theme: str, variant: int) -> Tuple[str, str]:
        """Generate notification title in English and Hinglish"""

        # These are domain-generic Octalysis-themed titles
        titles = {
            'accomplishment': [
                ("🎯 Keep the momentum!", "🎯 Momentum banaye rakho!"),
                ("🏆 You're doing great!", "🏆 Bahut badiya chal raha hai!"),
                ("⭐ Keep up the progress!", "⭐ Progress jaari rakho!"),
                ("🔥 You're on fire!", "🔥 Aag laga di tumne!"),
                ("💪 Keep crushing it!", "💪 Aise hi karo!")
            ],
            'loss_avoidance': [
                ("⚠️ Don't lose your progress!", "⚠️ Apna progress mat khoo!"),
                ("🔔 Your streak is at risk!", "🔔 Streak khatrey mein hai!"),
                ("⏰ Time is running out!", "⏰ Time khatam ho raha hai!"),
                ("❗ Your progress is at risk!", "❗ Progress khatrey mein!"),
                ("🚨 Act now!", "🚨 Abhi action lo!")
            ],
            'social_influence': [
                ("👥 See what others are doing!", "👥 Dekho baaki kya kar rahe!"),
                ("🏅 Leaderboard update!", "🏅 Leaderboard update!"),
                ("📊 You vs others", "📊 Tum vs Baaki log"),
                ("🌟 Join top performers!", "🌟 Top performers mein shamil ho!"),
                ("👋 Your peers are active!", "👋 Tumhare saathi active hain!")
            ],
            'unpredictability': [
                ("🎁 Surprise waiting!", "🎁 Surprise wait kar raha!"),
                ("✨ Something new for you!", "✨ Tumhare liye kuch naya!"),
                ("🎲 Mystery reward!", "🎲 Mystery reward!"),
                ("🔓 Unlock something special!", "🔓 Special cheez unlock karo!"),
                ("🎉 Discover today's bonus!", "🎉 Aaj ka bonus dekho!")
            ],
            'empowerment': [
                ("🎮 Your choice, your pace!", "🎮 Tumhari choice, tumhari speed!"),
                ("⚡ Take control!", "⚡ Control lo!"),
                ("🛠️ Customize your experience!", "🛠️ Apna experience customize karo!"),
                ("🎯 Do what you want!", "🎯 Jo chaho wo karo!"),
                ("💡 Your way!", "💡 Apne tarike se!")
            ],
            'ownership': [
                ("👑 Your achievements!", "👑 Tumhari achievements!"),
                ("💎 Your rewards are waiting!", "💎 Tumhare rewards wait kar rahe!"),
                ("🏠 Check your progress!", "🏠 Apna progress dekho!"),
                ("📈 Your stats update!", "📈 Tumhare stats ka update!"),
                ("🎖️ Your badges!", "🎖️ Tumhare badges!")
            ],
            'epic_meaning': [
                ("🚀 Join the movement!", "🚀 Movement mein shamil ho!"),
                ("🌍 Be part of something big!", "🌍 Kuch bada karo!"),
                ("💫 Transform your future!", "💫 Future transform karo!"),
                ("🎓 Your journey awaits!", "🎓 Tumhara safar shuru!"),
                ("✨ Make a difference!", "✨ Fark daalo!")
            ],
            'scarcity': [
                ("⏳ Limited time!", "⏳ Limited time!"),
                ("🔥 Offer ends soon!", "🔥 Offer jaldi khatam!"),
                ("⚡ Last chance today!", "⚡ Aaj ka last chance!"),
                ("🎯 Don't miss out!", "🎯 Miss mat karo!"),
                ("⏰ Today only!", "⏰ Sirf aaj!")
            ]
        }

        theme_titles = titles.get(theme, titles['accomplishment'])
        return theme_titles[variant % len(theme_titles)]

    def _generate_body(self, seg_name: str, lifecycle: str, goal: str,
                       theme: str, variant: int) -> Tuple[str, str]:
        """Generate notification body in English and Hinglish.

        Uses dynamic feature references from KB where appropriate.
        """
        # Pick a feature name to inject into bodies
        feature = self.feature_names[variant % len(self.feature_names)]

        bodies = {
            'accomplishment': [
                (f"You've made great progress this week. Keep it up to unlock your next achievement!",
                 f"Is hafte bahut progress hua. Aise hi karo aur next achievement unlock karo!"),
                ("Day {{streak_current}} of your streak! Complete today's session to keep going.",
                 "Streak ka din {{streak_current}}! Aaj ka session karo streak continue karne ke liye."),
                (f"Your {feature} usage is impressive—you're making amazing progress!",
                 f"Tumhara {feature} usage impressive hai—amazing progress ho raha hai!"),
                ("You've earned {{coins_balance}} coins! Complete another session to earn more.",
                 "Tumne {{coins_balance}} coins kamaye! Ek aur session karo aur coins badao."),
                ("Your {{streak_current}}-day streak is impressive! Keep it going today.",
                 "Tumhara {{streak_current}}-day streak amazing hai! Aaj bhi continue karo.")
            ],
            'loss_avoidance': [
                ("Your streak will break in {{hours_left}} hours! One quick session will save it.",
                 "Tumhara streak {{hours_left}} ghante mein toot jayega! Ek session se bach jayega."),
                ("Don't lose your {{streak_current}}-day streak! A quick session will save it.",
                 "Apna {{streak_current}}-day streak mat khoo! Ek chhoti session se bach jayega."),
                ("Your hard-earned progress is at risk. Complete today's session to protect it.",
                 "Tumhari mehnat khatrey mein hai. Aaj ka session karo protect karne ke liye."),
                ("Only {{hours_left}} hours left! Don't let your streak break.",
                 "Sirf {{hours_left}} ghante bache! Streak tootne mat do."),
                ("Your {{coins_balance}} coins will expire soon! Use them before it's too late.",
                 "Tumhare {{coins_balance}} coins expire hone wale! Jaldi use karo.")
            ],
            'social_influence': [
                (f"{{{{peer_name}}}} just completed a session on {feature}. Can you match that?",
                 f"{{{{peer_name}}}} ne {feature} par session complete kiya. Tum bhi kar sakte ho?"),
                ("Join {{active_users}}+ users active right now!",
                 "{{active_users}}+ users abhi active—tum bhi join karo!"),
                ("{{peer_name}} completed {{peer_exercises}} sessions today. Beat their score!",
                 "{{peer_name}} ne aaj {{peer_exercises}} sessions kiye. Unhe beat karo!"),
                ("You're #{{rank}} on the leaderboard. One session could push you higher!",
                 "Tum leaderboard par #{{rank}} ho. Ek session se aur upar jao!"),
                ("{{active_users}} users are online now—don't get left behind!",
                 "{{active_users}} users abhi online—peeche mat raho!")
            ],
            'unpredictability': [
                (f"A new {feature} experience has been unlocked just for you. Try it now!",
                 f"Ek naya {feature} experience sirf tumhare liye unlock hua. Abhi try karo!"),
                ("We have a surprise session waiting for you today. See what's in store!",
                 "Aaj tumhare liye surprise session hai. Dekho kya wait kar raha!"),
                ("Complete your next session to unlock a mystery reward!",
                 "Next session complete karo aur mystery reward unlock karo!"),
                ("Your score might surprise you today. Find out!",
                 "Tumhara score aaj surprise kar sakta hai. Dekho!"),
                (f"New {feature} scenarios just added! Explore them now.",
                 f"Naye {feature} scenarios add hue! Abhi explore karo.")
            ],
            'empowerment': [
                (f"Choose from 20+ options in {feature}—the choice is yours!",
                 f"Aaj {feature} mein 20+ options hai—choice tumhari!"),
                ("Go at your own pace. No pressure, just progress.",
                 "Apni speed se karo. Pressure nahi, sirf progress."),
                ("Customize your path. Pick what interests you most!",
                 "Apna path customize karo. Jo pasand ho wo choose karo!"),
                ("You control your progress. Start when you're ready.",
                 "Progress tumhare haath mein hai. Jab ready ho tab start karo."),
                (f"Pick any {feature} that interests you—it's your choice!",
                 f"Koi bhi {feature} choose karo—tumhari marzi!")
            ],
            'ownership': [
                ("Your {{coins_balance}} coins are ready to use! See what you can unlock.",
                 "Tumhare {{coins_balance}} coins ready hai! Dekho kya unlock kar sakte ho."),
                ("You've built a {{streak_current}}-day streak! That's YOUR achievement.",
                 "Tumne {{streak_current}}-day streak banaya! Ye TUMHARI achievement hai."),
                ("Your progress dashboard has updates. Check out YOUR stats!",
                 "Tumhare progress dashboard mein updates. APNE stats dekho!"),
                ("You've earned {{coins_balance}} coins—use them to unlock premium features!",
                 "Tumne {{coins_balance}} coins kamaye—premium features unlock karo!"),
                ("Your journey has been impressive. Keep building YOUR story!",
                 "Tumhara journey impressive raha. APNI kahani banao!")
            ],
            'epic_meaning': [
                ("Join 1M+ users who are transforming their lives!",
                 "10 lakh+ users join karo jo apni life transform kar rahe!"),
                ("Be part of the transformation revolution!",
                 "Transformation revolution ka hissa bano!"),
                ("Transform your future with better skills. Start today!",
                 "Apna future transform karo better skills se. Aaj start karo!"),
                ("Join thousands who improved this month!",
                 "Hazaron logon ke saath judo jinhone is mahine improve kiya!"),
                (f"Your journey with {feature} starts with one session today.",
                 f"{feature} ke saath safar aaj ek session se shuru karo.")
            ],
            'scarcity': [
                ("Only 3 hours left to complete today's goal. Don't wait!",
                 "Aaj ka goal complete karne ke liye sirf 3 ghante bache. Jaldi karo!"),
                ("Limited time offer: Double coins on your next session!",
                 "Limited time offer: Next session par double coins!"),
                ("Today's special content expires in 2 hours. Try it now!",
                 "Aaj ka special content 2 ghante mein expire. Abhi try karo!"),
                ("Last chance to maintain your streak today. Act now!",
                 "Aaj streak maintain karne ka last chance. Abhi action lo!"),
                ("This opportunity expires at midnight. Don't miss it!",
                 "Ye opportunity midnight ko expire. Miss mat karo!")
            ]
        }

        theme_bodies = bodies.get(theme, bodies['accomplishment'])
        return theme_bodies[variant % len(theme_bodies)]

    def _generate_cta(self, goal: str, theme: str, variant: int) -> Tuple[str, str]:
        """Generate CTA button text in English and Hinglish"""

        ctas = {
            'accomplishment': [
                ("Continue Now", "Abhi Continue Karo"),
                ("Start Session", "Session Shuru Karo"),
                ("Keep Going", "Aage Badho"),
                ("Complete Now", "Abhi Complete Karo"),
                ("Earn More Coins", "Aur Coins Kamao")
            ],
            'loss_avoidance': [
                ("Save My Streak", "Mera Streak Bachao"),
                ("Act Now", "Abhi Karo"),
                ("Don't Lose It", "Kho Mat"),
                ("Protect Progress", "Progress Protect Karo"),
                ("Save Now", "Abhi Bachao")
            ],
            'social_influence': [
                ("Join Now", "Abhi Join Karo"),
                ("Beat Them", "Unhe Harao"),
                ("See Leaderboard", "Leaderboard Dekho"),
                ("Compete Now", "Abhi Compete Karo"),
                ("Climb Higher", "Upar Jao")
            ],
            'unpredictability': [
                ("Discover Now", "Abhi Discover Karo"),
                ("See Surprise", "Surprise Dekho"),
                ("Unlock It", "Unlock Karo"),
                ("Try It", "Try Karo"),
                ("Find Out", "Pata Karo")
            ],
            'empowerment': [
                ("Choose Now", "Abhi Choose Karo"),
                ("Start My Way", "Apne Tarike Se Start"),
                ("Customize", "Customize Karo"),
                ("Pick Activity", "Activity Choose Karo"),
                ("My Choice", "Meri Choice")
            ],
            'ownership': [
                ("View My Stats", "Mere Stats Dekho"),
                ("Use My Coins", "Mere Coins Use Karo"),
                ("See Progress", "Progress Dekho"),
                ("My Dashboard", "Mera Dashboard"),
                ("Check Rewards", "Rewards Check Karo")
            ],
            'epic_meaning': [
                ("Join Movement", "Movement Join Karo"),
                ("Start Journey", "Safar Shuru Karo"),
                ("Transform Now", "Abhi Transform Karo"),
                ("Be Part Of It", "Iska Hissa Bano"),
                ("Begin Today", "Aaj Shuru Karo")
            ],
            'scarcity': [
                ("Grab Now", "Abhi Pakdo"),
                ("Don't Wait", "Wait Mat Karo"),
                ("Hurry Up", "Jaldi Karo"),
                ("Claim Offer", "Offer Claim Karo"),
                ("Last Chance", "Last Chance")
            ]
        }

        theme_ctas = ctas.get(theme, ctas['accomplishment'])
        return theme_ctas[variant % len(theme_ctas)]

    def _generate_content(self, seg_name: str, lifecycle: str, goal: str,
                         theme: str, variant: int) -> str:
        """Generate template content based on parameters (legacy support)"""
        _, body_en = self._generate_body(seg_name, lifecycle, goal, theme, variant)
        return body_en

    def _select_tone(self, lifecycle: str, theme: str) -> str:
        """Select appropriate tone"""
        tone_map = {
            'trial': {
                'accomplishment': 'encouraging',
                'loss_avoidance': 'urgent',
                'social_influence': 'motivational',
                'unpredictability': 'inviting',
                'empowerment': 'supportive',
                'ownership': 'celebratory',
                'epic_meaning': 'aspirational',
                'scarcity': 'urgent'
            },
            'paid': {
                'accomplishment': 'celebratory',
                'loss_avoidance': 'friendly',
                'social_influence': 'motivational',
                'unpredictability': 'inviting',
                'empowerment': 'supportive',
                'ownership': 'celebratory',
                'epic_meaning': 'aspirational',
                'scarcity': 'friendly'
            },
            'churned': {
                'accomplishment': 'encouraging',
                'loss_avoidance': 'urgent',
                'social_influence': 'motivational',
                'unpredictability': 'curious',
                'empowerment': 'supportive',
                'ownership': 'friendly',
                'epic_meaning': 'aspirational',
                'scarcity': 'urgent'
            },
            'inactive': {
                'accomplishment': 'encouraging',
                'loss_avoidance': 'friendly',
                'social_influence': 'inviting',
                'unpredictability': 'curious',
                'empowerment': 'supportive',
                'ownership': 'friendly',
                'epic_meaning': 'aspirational',
                'scarcity': 'friendly'
            }
        }

        return tone_map.get(lifecycle, {}).get(theme, 'friendly')

    def _select_feature(self, goal: str, theme: str) -> str:
        """Select relevant feature reference from KB-extracted features."""
        # Map goals to likely feature types
        goal_feature_hints = {
            'activation': ['core', 'onboard', 'start', 'first'],
            'habit_formation': ['streak', 'daily', 'habit', 'track'],
            'feature_discovery': ['tutor', 'ai', 'new', 'explore'],
            'conversion_readiness': ['premium', 'subscribe', 'value'],
            'retention': ['core', 'engage', 'main'],
            'expansion': ['leaderboard', 'social', 'compete', 'community'],
            'advocacy': ['share', 'refer', 'invite'],
            're_engagement': ['new', 'update', 'fresh'],
            'skill_development': ['learn', 'practice', 'tutor'],
        }

        hints = goal_feature_hints.get(goal, [])

        # Try to match a KB feature name to the goal hints
        for fname in self.feature_names:
            fname_lower = fname.lower()
            for hint in hints:
                if hint in fname_lower:
                    return fname

        # Fallback to first feature
        return self.feature_names[0] if self.feature_names else 'Core Feature'

    def save_templates(self, output_dir: str):
        """Save templates to CSV"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        self.templates.to_csv(output_path / 'message_templates.csv', index=False)

        print(f"[OK] Templates saved to {output_dir}/message_templates.csv")
